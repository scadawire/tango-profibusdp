# https://pypi.org/project/pyprofibus/

import os
import json
import struct
import traceback
import threading
from tango import AttrWriteType, AttrDataFormat, DevState, Attr, SpectrumAttr, ImageAttr, CmdArgType, UserDefaultAttrProp
from tango.server import Device, attribute, command, DeviceMeta, device_property, run
# pyprofibus is imported lazily inside connect() so the module can be loaded
# (and tested) without the library installed.


class ProfibusDp(Device, metaclass=DeviceMeta):

    # ───────────── Device Properties ─────────────
    serial_port = device_property(dtype=str, default_value="/dev/ttyUSB0")
    baudrate = device_property(dtype=int, default_value=19200)
    master_addr = device_property(dtype=int, default_value=2)
    # JSON array: [{"addr": 8, "input_size": 4, "output_size": 4, "ident": 0}]
    slave_configs = device_property(dtype=str, default_value="[]")
    cycle_time = device_property(dtype=float, default_value=0.01)
    init_dynamic_attributes = device_property(dtype=str, default_value="")

    # ───────────── Lifecycle ─────────────
    def init_device(self):
        self.set_state(DevState.INIT)
        self.get_device_properties(self.get_device_class())
        self._slave_input_data = {}   # slave_addr -> bytearray  (slave → master)
        self._slave_output_data = {}  # slave_addr -> bytearray  (master → slave)
        self._slave_descs = {}        # slave_addr -> DpSlaveDesc
        self.dynamicAttributes = {}
        self.dynamicAttributeLookup = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.last_error = ""
        self._master = None

        self.connect()

        if self.init_dynamic_attributes:
            try:
                attrs = json.loads(self.init_dynamic_attributes)
                for a in attrs:
                    self.add_dynamic_attribute(
                        a["name"],
                        a.get("data_type", ""),
                        a.get("min_value", ""),
                        a.get("max_value", ""),
                        a.get("unit", ""),
                        a.get("write_type", ""),
                        a.get("label", ""),
                        a.get("register", ""),
                        a.get("data_format", ""),
                        str(a.get("max_x", "")),
                        str(a.get("max_y", "")),
                    )
            except Exception:
                for name in self.init_dynamic_attributes.split(","):
                    self.add_dynamic_attribute(name.strip())

        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self.set_state(DevState.ON)

    def delete_device(self):
        if hasattr(self, "_stop_event"):
            self._stop_event.set()

    # ───────────── Connection ─────────────
    def connect(self):
        try:
            from pyprofibus.phy_serial import CpPhySerial
            from pyprofibus import DPM1

            phy = CpPhySerial(port=self.serial_port, debug=False)
            phy.setConfig(baudrate=self.baudrate)

            self._master = DPM1(
                phy=phy,
                masterAddr=self.master_addr,
                debug=False,
            )

            for sc in json.loads(self.slave_configs):
                addr = int(sc["addr"])
                input_size = int(sc.get("input_size", 0))
                output_size = int(sc.get("output_size", 0))
                ident = int(sc.get("ident", 0))

                slaveDesc = pyprofibus.DpSlaveDesc(
                    identNumber=ident,
                    slaveAddr=addr,
                    inputSize=input_size,
                    outputSize=output_size,
                )
                self._slave_input_data[addr] = bytearray(input_size)
                self._slave_output_data[addr] = bytearray(output_size)
                self._slave_descs[addr] = slaveDesc
                self._master.addSlave(slaveDesc)

            self._master.initialize()
            self.info_stream("Connected to Profibus DP master on %s", self.serial_port)

        except Exception as e:
            self.last_error = str(e)
            self.error_stream("%s", traceback.format_exc())
            self.set_state(DevState.FAULT)

    # ───────────── Poll Loop ─────────────
    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                self._run_cycle()
            except Exception as e:
                self.warn_stream("poll error: %s", str(e))
            self._stop_event.wait(timeout=self.cycle_time)

    def _run_cycle(self):
        if self._master is None:
            return

        # Push current output buffer to every slave before stepping the master.
        for addr, slaveDesc in self._slave_descs.items():
            with self._lock:
                out = bytes(self._slave_output_data[addr])
            slaveDesc.setMasterOutData(out)

        # master.run() handles ONE slave per call (round-robin) and returns its
        # DpSlaveDesc, or None if nothing was processed this step.
        handledDesc = self._master.run()
        if handledDesc is None:
            return

        data = handledDesc.getMasterInData()
        if data is None:
            return

        # Update input cache for the slave that was just serviced.
        for addr, slaveDesc in self._slave_descs.items():
            if slaveDesc is handledDesc:
                with self._lock:
                    size = len(self._slave_input_data[addr])
                    self._slave_input_data[addr][:len(data)] = data[:size]
                break

    # ───────────── Dynamic Attributes ─────────────
    def add_dynamic_attribute(
        self, name, variable_type_name="DevString",
        min_value="", max_value="", unit="",
        write_type_name="", label="", register="",
        data_format_name="", max_x="", max_y=""
    ):
        if not name:
            return

        prop = UserDefaultAttrProp()
        var_type = self.stringValueToVarType(variable_type_name)
        reg = self.parse_register(register)
        data_format = self.stringValueToFormatType(data_format_name)
        dim_x = int(max_x) if max_x else 256
        dim_y = int(max_y) if max_y else 256

        # Input (slave→master) is read-only by default
        if write_type_name == "" and reg["io_type"] == "i":
            effective_write_type_name = "READ"
        else:
            effective_write_type_name = write_type_name
        write_type = self.stringValueToWriteType(effective_write_type_name)

        if unit:
            prop.set_unit(unit)
        if label:
            prop.set_label(label)
        if min_value:
            prop.set_min_value(min_value)
        if max_value:
            prop.set_max_value(max_value)

        if data_format == AttrDataFormat.SPECTRUM:
            attr = SpectrumAttr(name, var_type, write_type, dim_x)
        elif data_format == AttrDataFormat.IMAGE:
            attr = ImageAttr(name, var_type, write_type, dim_x, dim_y)
        else:
            attr = Attr(name, var_type, write_type)
        attr.set_default_properties(prop)

        self.dynamicAttributeLookup[name] = {
            "variableType": var_type,
            "slave_addr": reg["slave_addr"],
            "io_type": reg["io_type"],
            "byte_offset": reg["byte_offset"],
            "suboffset": reg["suboffset"],
            "dataFormat": data_format,
            "max_x": dim_x,
            "max_y": dim_y,
        }
        self.dynamicAttributes[name] = 0
        self.add_attribute(attr, r_meth=self.read_dynamic_attr, w_meth=self.write_dynamic_attr)
        self.set_change_event(name, True, False)

    # ───────────── Attribute Access ─────────────
    def read_dynamic_attr(self, attr):
        name = attr.get_name()
        lookup = self.dynamicAttributeLookup[name]
        data_format = lookup["dataFormat"]

        if data_format == AttrDataFormat.SPECTRUM:
            value = self._read_array(lookup, lookup["max_x"])
        elif data_format == AttrDataFormat.IMAGE:
            flat = self._read_array(lookup, lookup["max_x"] * lookup["max_y"])
            value = [flat[r * lookup["max_x"]:(r + 1) * lookup["max_x"]] for r in range(lookup["max_y"])]
        else:
            value = self._read_scalar(lookup)

        attr.set_value(value)

    def write_dynamic_attr(self, attr):
        name = attr.get_name()
        value = attr.get_write_value()
        lookup = self.dynamicAttributeLookup[name]
        data_format = lookup["dataFormat"]

        if data_format == AttrDataFormat.SPECTRUM:
            self._write_array(lookup, value)
        elif data_format == AttrDataFormat.IMAGE:
            flat = [v for row in value for v in row]
            self._write_array(lookup, flat)
        else:
            self._write_scalar(lookup, value)
        self.push_change_event(name, value)

    def _read_scalar(self, lookup):
        slave_addr = lookup["slave_addr"]
        io_type = lookup["io_type"]
        byte_offset = lookup["byte_offset"]
        suboffset = lookup["suboffset"]
        var_type = lookup["variableType"]

        with self._lock:
            buf = self._slave_input_data[slave_addr] if io_type == "i" else self._slave_output_data[slave_addr]
            n = self._bytes_for_type(var_type, suboffset)
            raw = bytes(buf[byte_offset:byte_offset + n])
        return self.bytedata_to_variable(raw, var_type, suboffset)

    def _write_scalar(self, lookup, value):
        slave_addr = lookup["slave_addr"]
        byte_offset = lookup["byte_offset"]
        suboffset = lookup["suboffset"]
        var_type = lookup["variableType"]

        if var_type == CmdArgType.DevBoolean:
            self._rmw_bool(slave_addr, byte_offset, suboffset, value)
            return

        raw = self.variable_to_bytedata(value, var_type, suboffset)
        with self._lock:
            buf = self._slave_output_data[slave_addr]
            buf[byte_offset:byte_offset + len(raw)] = raw

    def _rmw_bool(self, slave_addr, byte_offset, bit_index, value):
        """Read-modify-write a single bit in the output buffer (bit_index 0-7)."""
        with self._lock:
            buf = self._slave_output_data[slave_addr]
            if value:
                buf[byte_offset] |= (1 << bit_index)
            else:
                buf[byte_offset] &= ~(1 << bit_index)

    def _read_array(self, lookup, count):
        slave_addr = lookup["slave_addr"]
        io_type = lookup["io_type"]
        byte_offset = lookup["byte_offset"]
        var_type = lookup["variableType"]
        elem_size = self.bytes_per_variable_type(var_type)

        with self._lock:
            buf = self._slave_input_data[slave_addr] if io_type == "i" else self._slave_output_data[slave_addr]
            raw = bytes(buf[byte_offset:byte_offset + elem_size * count])

        return [
            self.bytedata_to_variable(raw[i * elem_size:(i + 1) * elem_size], var_type)
            for i in range(count)
        ]

    def _write_array(self, lookup, values):
        slave_addr = lookup["slave_addr"]
        byte_offset = lookup["byte_offset"]
        var_type = lookup["variableType"]

        raw = b"".join(self.variable_to_bytedata(v, var_type) for v in values)
        with self._lock:
            buf = self._slave_output_data[slave_addr]
            buf[byte_offset:byte_offset + len(raw)] = raw

    # ───────────── Register Parsing ─────────────
    _IO_TYPE_MAP = {
        "i": "i", "input": "i",
        "o": "o", "output": "o",
    }

    def parse_register(self, register):
        try:
            parts = register.split(".")
            slave_addr = int(parts[0])
            io_type = self._IO_TYPE_MAP.get(parts[1].lower())
            if io_type is None:
                raise ValueError(f"Unknown io_type '{parts[1]}'")
            byte_offset = int(parts[2], 0)
            suboffset = int(parts[3]) if len(parts) > 3 else 0
            return {
                "slave_addr": slave_addr,
                "io_type": io_type,
                "byte_offset": byte_offset,
                "suboffset": suboffset,
            }
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid register descriptor '{register}', "
                f"expected: slave_addr.io_type.byte_offset[.suboffset]"
            )

    # ───────────── Type Helpers ─────────────
    def _bytes_for_type(self, var_type, suboffset=0):
        if var_type == CmdArgType.DevString:
            return suboffset
        return self.bytes_per_variable_type(var_type)

    def bytes_per_variable_type(self, var_type):
        return {
            CmdArgType.DevBoolean: 1,
            CmdArgType.DevShort:   2,
            CmdArgType.DevFloat:   4,
            CmdArgType.DevLong:    4,
            CmdArgType.DevDouble:  8,
            CmdArgType.DevLong64:  8,
        }.get(var_type, 0)

    def bytedata_to_variable(self, data, var_type, suboffset=0):
        if var_type == CmdArgType.DevBoolean:
            if isinstance(data, (bytes, bytearray)):
                return bool((data[0] >> suboffset) & 0x01)
            return bool(data)
        elif var_type == CmdArgType.DevFloat:
            return struct.unpack(">f", data[:4])[0]
        elif var_type == CmdArgType.DevDouble:
            return struct.unpack(">d", data[:8])[0]
        elif var_type == CmdArgType.DevLong:
            return struct.unpack(">i", data[:4])[0]
        elif var_type == CmdArgType.DevShort:
            return struct.unpack(">h", data[:2])[0]
        elif var_type == CmdArgType.DevLong64:
            return struct.unpack(">q", data[:8])[0]
        elif var_type == CmdArgType.DevString:
            if isinstance(data, (bytes, bytearray)):
                end = data.find(b"\x00")
                if end == -1:
                    end = len(data)
                return data[:end].decode("utf-8", errors="ignore")
            return str(data)
        else:
            raise ValueError(f"Unsupported variable type: {var_type}")

    def variable_to_bytedata(self, value, var_type, suboffset=0):
        if var_type == CmdArgType.DevBoolean:
            return struct.pack("B", 1 if value else 0)
        elif var_type == CmdArgType.DevFloat:
            return struct.pack(">f", float(value))
        elif var_type == CmdArgType.DevDouble:
            return struct.pack(">d", float(value))
        elif var_type == CmdArgType.DevLong:
            return struct.pack(">i", int(value))
        elif var_type == CmdArgType.DevShort:
            return struct.pack(">h", int(value))
        elif var_type == CmdArgType.DevLong64:
            return struct.pack(">q", int(value))
        elif var_type == CmdArgType.DevString:
            raw = str(value).encode("utf-8")
            if len(raw) >= suboffset:
                raise ValueError(f"String too long to fit into {suboffset} bytes")
            raw += b"\x00" * (suboffset - len(raw))
            return raw
        else:
            raise ValueError(f"Unsupported variable type: {var_type}")

    def stringValueToVarType(self, name):
        return {
            "DevBoolean": CmdArgType.DevBoolean,
            "DevShort":   CmdArgType.DevShort,
            "DevLong":    CmdArgType.DevLong,
            "DevFloat":   CmdArgType.DevFloat,
            "DevDouble":  CmdArgType.DevDouble,
            "DevLong64":  CmdArgType.DevLong64,
            "DevString":  CmdArgType.DevString,
            "":           CmdArgType.DevString,
        }.get(name, CmdArgType.DevString)

    def stringValueToWriteType(self, name):
        return {
            "READ":       AttrWriteType.READ,
            "WRITE":      AttrWriteType.WRITE,
            "READ_WRITE": AttrWriteType.READ_WRITE,
            "":           AttrWriteType.READ_WRITE,
        }.get(name, AttrWriteType.READ_WRITE)

    def stringValueToFormatType(self, name):
        return {
            "SCALAR":   AttrDataFormat.SCALAR,
            "SPECTRUM": AttrDataFormat.SPECTRUM,
            "IMAGE":    AttrDataFormat.IMAGE,
        }.get(name, AttrDataFormat.SCALAR)


if __name__ == "__main__":
    server_name = os.getenv("DEVICE_SERVER_NAME", "ProfibusDp")
    run({server_name: ProfibusDp})
