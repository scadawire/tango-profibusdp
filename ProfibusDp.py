# https://pypi.org/project/pyprofibus/

import os
import json
import shutil
import struct
import tempfile
import traceback
import threading
from tango import AttrWriteType, AttrDataFormat, DevState, Attr, SpectrumAttr, ImageAttr, CmdArgType, UserDefaultAttrProp
from tango.server import Device, attribute, command, DeviceMeta, device_property, run
# pyprofibus is imported lazily inside connect() so the module can be loaded
# (and tested) without the library installed.


class ProfibusDp(Device, metaclass=DeviceMeta):

    # ───────────── Device Properties ─────────────
    # Which CP-PHY carries the bus: "serial" (CpPhySerial on serial_port),
    # "dummy_slave" (CpPhyDummySlave, an in-process slave that echoes every
    # DataExchange byte XOR 0xFF - used by the SITL test) or "fpga".
    # The values are passed through to pyprofibus, see PbConf.makePhy().
    phy_type = device_property(dtype=str, default_value="serial")
    serial_port = device_property(dtype=str, default_value="/dev/ttyUSB0")
    baudrate = device_property(dtype=int, default_value=19200)
    master_addr = device_property(dtype=int, default_value=2)
    # JSON array, one entry per slave:
    #   [{"addr": 8, "gsd_file": "#Profibus_DP\nGSD_Revision=1\n...",
    #     "modules": ["dummy output module", "dummy input module"],
    #     "input_size": 2, "output_size": 2}]
    # gsd_file carries the CONTENT of the slave's gsd, not a path to it - the
    # same way the canopen device server takes its eds_file.  Nobody can put a
    # file on this machine: scadawire is configured over the web, so a path
    # property would be unusable.  connect() writes the content out to a temp
    # file only because pyprofibus parses a gsd from a filename and nothing else.
    # A gsd is mandatory: pyprofibus derives the ident number, the Chk_Cfg data
    # elements and the User_Prm_Data from it, and none of them can be recovered
    # from sizes alone.  input_size/output_size are named from the SLAVE's point
    # of view, as in the gsd and in pyprofibus - input_size is what the slave
    # receives from us.  Optional per slave: name, sync_mode, freeze_mode,
    # group_mask, watchdog_ms, diag_period.
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

        connected = self.connect()

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
        # only claim ON when there is a master to be on: connect() puts us in FAULT when it fails,
        # and setting ON here regardless would paper over it. The slave buffers are empty then, so
        # the first attribute access is what would surface the failure, as a KeyError on the slave
        # address rather than as the connection error it is.
        if connected:
            self.set_state(DevState.ON)

    def delete_device(self):
        if hasattr(self, "_stop_event"):
            self._stop_event.set()

    # ───────────── Connection ─────────────
    # Everything is built from a rendered pyprofibus conf rather than by calling
    # DpSlaveDesc/DPM1 directly.  That is not a detour: DpSlaveDesc(slaveConf)
    # takes a PbConf slave section and nothing else, and the Set_Prm / Chk_Cfg
    # telegrams it prepares are filled in by PbConf._SlaveConf.makeDpSlaveDesc()
    # from the gsd.  Going through PbConf also buys the phy switch for free -
    # PbConf.makePhy() dispatches on [PHY] type and sizes the dummy slave's echo
    # from the slave configs.
    # Returns whether a master was built and initialized, so init_device knows whether it may go ON.
    def connect(self):
        try:
            from pyprofibus import PbConf

            # The conf and the gsd of every slave only exist to be parsed: pyprofibus reads both
            # from a filename, while the properties carry their content. fromFile() parses the gsd
            # of every slave section eagerly, so once it returns nothing needs the directory again.
            temp_dir = tempfile.mkdtemp(prefix="profibusdp-")
            try:
                pbConf = PbConf.fromFile(self.render_conf(temp_dir))
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            self._master = pbConf.makeDPM()

            for slaveConf in pbConf.slaveConfs:
                slaveDesc = slaveConf.makeDpSlaveDesc()
                self._master.addSlave(slaveDesc)

                # pyprofibus names the sizes from the slave's point of view, we
                # name our buffers from the master's: the slave's output is our
                # input.  Getting this crossover wrong truncates silently, since
                # _run_cycle slices the incoming data to the buffer length.
                self._slave_descs[slaveConf.addr] = slaveDesc
                self._slave_input_data[slaveConf.addr] = bytearray(slaveConf.outputSize)
                self._slave_output_data[slaveConf.addr] = bytearray(slaveConf.inputSize)

            self._master.initialize()
            self.info_stream(
                "Connected to Profibus DP master, phy %s on %s",
                self.phy_type, self.serial_port
            )
            return True

        except Exception as e:
            self.last_error = str(e)
            self.error_stream("%s", traceback.format_exc())
            self.set_state(DevState.FAULT)
            return False

    def render_conf(self, temp_dir):
        """
        Render the device properties into a pyprofibus conf file in temp_dir, next to the gsd of
        every slave. Returns the path of the conf; the caller owns temp_dir and has to remove it.
        """
        lines = [
            "[PROFIBUS]",
            "debug=0",
            "",
            "[PHY]",
            "type=%s" % self.phy_type,
            "dev=%s" % self.serial_port,
            "baud=%d" % self.baudrate,
            "",
            "[FDL]",
            "",
            "[DP]",
            "master_class=1",
            "master_addr=%d" % self.master_addr,
            "",
        ]

        for index, sc in enumerate(json.loads(self.slave_configs)):
            lines.append("[SLAVE_%d]" % index)
            lines.append("name=%s" % sc.get("name", "slave_%d" % index))
            lines.append("addr=%d" % int(sc["addr"]))
            lines.append("gsd=%s" % self.write_gsd(temp_dir, index, sc))
            lines.append("sync_mode=%d" % (1 if sc.get("sync_mode") else 0))
            lines.append("freeze_mode=%d" % (1 if sc.get("freeze_mode") else 0))
            lines.append("group_mask=%d" % int(sc.get("group_mask", 1)))
            lines.append("watchdog_ms=%d" % int(sc.get("watchdog_ms", 5000)))
            # Modules are only meaningful for a modular station, and have to be
            # listed in slot order: pyprofibus feeds them to the gsd in the order
            # of the module_N keys to build the cfg data elements.
            for mod_index, module in enumerate(sc.get("modules", [])):
                lines.append("module_%d=%s" % (mod_index, module))
            lines.append("input_size=%d" % int(sc.get("input_size", 0)))
            lines.append("output_size=%d" % int(sc.get("output_size", 0)))
            lines.append("diag_period=%d" % int(sc.get("diag_period", 0)))
            lines.append("")

        path = os.path.join(temp_dir, "profibus.conf")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    def write_gsd(self, temp_dir, index, slave_config):
        """
        Write the slave's gsd content out into temp_dir and return the path to hand to pyprofibus.
        """
        content = slave_config.get("gsd_file", "")
        if not content.strip():
            raise ValueError(
                "slave_configs[%d] (addr %s) has no gsd_file. It carries the content of the "
                "slave's gsd, which pyprofibus needs for the ident number, the Chk_Cfg data and "
                "the User_Prm_Data - none of which can be derived from the sizes"
                % (index, slave_config.get("addr"))
            )
        path = os.path.join(temp_dir, "slave_%d.gsd" % index)
        with open(path, "w") as f:
            f.write(content)
        return path

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
