"""
Full unit test for ProfibusDp.

Two test categories:

1. Buffer tests (no hardware / no PHY):
   All variable types (DevBoolean, DevShort, DevLong, DevFloat, DevDouble,
   DevLong64, DevString) are exercised against in-memory byte buffers via the
   real ProfibusDp class using unbound method calls.

2. Simulator tests (CpPhyDummySlave):
   Uses pyprofibus's built-in dummy PHY to run the full Profibus DP state
   machine in-process.  The dummy slave echoes all output bytes XOR 0xFF back
   as input bytes, so value round-trips are tested by pre-inverting the
   expected encoding before writing to the output buffer.

   Requires: pyprofibus installed, dummy_modular.gsd in the same directory.
   Skipped automatically if pyprofibus is not installed.

Usage:
    python test_profibusdp.py
"""

import os
import struct
import tempfile
import threading
import sys
import traceback

from tango import CmdArgType, AttrDataFormat

from ProfibusDp import ProfibusDp


# ===========================================================================
#  Lightweight state carrier
# ===========================================================================

class State:
    """Carries only instance state; every method lookup falls through to ProfibusDp."""
    _IO_TYPE_MAP = ProfibusDp._IO_TYPE_MAP

    def __init__(self):
        self._slave_input_data = {}
        self._slave_output_data = {}
        self._slave_descs = {}
        self.dynamicAttributeLookup = {}
        self._lock = threading.Lock()

    def setup_slave(self, addr, input_size, output_size):
        self._slave_input_data[addr] = bytearray(input_size)
        self._slave_output_data[addr] = bytearray(output_size)

    # mirror pytango: the message is rendered as `msg % args`, so a format
    # mismatch (a stray % in the payload) raises here just as it would live
    def info_stream(self, msg, *args): msg % args
    def debug_stream(self, msg, *args): msg % args
    def warn_stream(self, msg, *args): msg % args
    def error_stream(self, msg, *args): msg % args

    def __getattr__(self, name):
        attr = getattr(ProfibusDp, name, None)
        if attr is not None and callable(attr):
            import functools
            return functools.partial(attr, self)
        raise AttributeError(f"'State' has no attribute '{name}'")


# ===========================================================================
#  Thin helpers
# ===========================================================================

def register_attr(s, name, register_str, var_type,
                  data_format=AttrDataFormat.SCALAR, max_x=0, max_y=0):
    s.dynamicAttributeLookup[name] = {
        **ProfibusDp.parse_register(s, register_str),
        "variableType": var_type,
        "dataFormat": data_format,
        "max_x": max_x,
        "max_y": max_y,
    }


def read_attr(s, name):
    lookup = s.dynamicAttributeLookup[name]
    data_format = lookup["dataFormat"]
    if data_format == AttrDataFormat.SPECTRUM:
        return ProfibusDp._read_array(s, lookup, lookup["max_x"])
    elif data_format == AttrDataFormat.IMAGE:
        flat = ProfibusDp._read_array(s, lookup, lookup["max_x"] * lookup["max_y"])
        return [flat[r * lookup["max_x"]:(r + 1) * lookup["max_x"]] for r in range(lookup["max_y"])]
    else:
        return ProfibusDp._read_scalar(s, lookup)


def write_attr(s, name, value):
    lookup = s.dynamicAttributeLookup[name]
    data_format = lookup["dataFormat"]
    if data_format == AttrDataFormat.SPECTRUM:
        ProfibusDp._write_array(s, lookup, value)
    elif data_format == AttrDataFormat.IMAGE:
        flat = [v for row in value for v in row]
        ProfibusDp._write_array(s, lookup, flat)
    else:
        ProfibusDp._write_scalar(s, lookup, value)


# ===========================================================================
#  Test helpers
# ===========================================================================

passed = 0
failed = 0
errors = []


def assert_equal(test_name, actual, expected, tolerance=None):
    global passed, failed
    ok = abs(actual - expected) <= tolerance if tolerance is not None else (actual == expected)
    if ok:
        passed += 1
        print(f"  PASS  {test_name}")
    else:
        failed += 1
        msg = f"  FAIL  {test_name}: expected {expected!r}, got {actual!r}"
        print(msg)
        errors.append(msg)


def assert_list_equal(test_name, actual, expected, tolerance=None):
    global passed, failed
    if len(actual) == len(expected):
        if tolerance is not None:
            ok = all(abs(a - e) <= tolerance for a, e in zip(actual, expected))
        else:
            ok = (actual == expected)
    else:
        ok = False
    if ok:
        passed += 1
        print(f"  PASS  {test_name}")
    else:
        failed += 1
        msg = f"  FAIL  {test_name}: expected {expected!r}, got {actual!r}"
        print(msg)
        errors.append(msg)


def assert_2d_equal(test_name, actual, expected, tolerance=None):
    global passed, failed
    ok = len(actual) == len(expected)
    if ok:
        for row_a, row_e in zip(actual, expected):
            if len(row_a) != len(row_e):
                ok = False
                break
            if tolerance is not None:
                if not all(abs(a - e) <= tolerance for a, e in zip(row_a, row_e)):
                    ok = False
                    break
            elif row_a != row_e:
                ok = False
                break
    if ok:
        passed += 1
        print(f"  PASS  {test_name}")
    else:
        failed += 1
        msg = f"  FAIL  {test_name}: expected {expected!r}, got {actual!r}"
        print(msg)
        errors.append(msg)


def assert_true(test_name, value):
    assert_equal(test_name, value, True)


def assert_false(test_name, value):
    assert_equal(test_name, value, False)


def assert_raises(test_name, fn, *args):
    global passed, failed
    try:
        fn(*args)
        failed += 1
        msg = f"  FAIL  {test_name}: expected exception, none raised"
        print(msg)
        errors.append(msg)
    except Exception:
        passed += 1
        print(f"  PASS  {test_name}")


# ===========================================================================
#  Test suites
# ===========================================================================

def test_parse_register():
    print("\n-- parse_register --")
    s = State()

    r = ProfibusDp.parse_register(s, "8.i.0")
    assert_equal("slave_addr", r["slave_addr"], 8)
    assert_equal("io_type input", r["io_type"], "i")
    assert_equal("byte_offset", r["byte_offset"], 0)
    assert_equal("suboffset default", r["suboffset"], 0)

    r = ProfibusDp.parse_register(s, "8.o.4.3")
    assert_equal("io_type output", r["io_type"], "o")
    assert_equal("byte_offset 4", r["byte_offset"], 4)
    assert_equal("suboffset 3", r["suboffset"], 3)

    r = ProfibusDp.parse_register(s, "12.input.0x08")
    assert_equal("long-form input", r["io_type"], "i")
    assert_equal("hex offset", r["byte_offset"], 8)

    r = ProfibusDp.parse_register(s, "3.output.16.7")
    assert_equal("long-form output", r["io_type"], "o")
    assert_equal("bit 7 suboffset", r["suboffset"], 7)

    assert_raises("invalid register", ProfibusDp.parse_register, s, "bad.x.y")
    assert_raises("unknown io_type", ProfibusDp.parse_register, s, "8.z.0")


def test_byte_conversions():
    print("\n-- byte conversions (round-trip) --")
    s = State()

    # DevFloat
    for val in [0.0, 1.0, -1.0, 3.14159, 1e10, -1e-5]:
        enc = ProfibusDp.variable_to_bytedata(s, val, CmdArgType.DevFloat)
        dec = ProfibusDp.bytedata_to_variable(s, enc, CmdArgType.DevFloat)
        assert_equal(f"DevFloat {val}", dec, val, tolerance=max(abs(val) * 1e-6, 1e-7))

    # DevDouble
    for val in [0.0, 2.718281828459045, -1e100, 1e-300]:
        enc = ProfibusDp.variable_to_bytedata(s, val, CmdArgType.DevDouble)
        dec = ProfibusDp.bytedata_to_variable(s, enc, CmdArgType.DevDouble)
        assert_equal(f"DevDouble {val}", dec, val, tolerance=max(abs(val) * 1e-12, 1e-300))

    # DevLong
    for val in [0, 1, -1, 32767, -32768, 2147483647, -2147483648]:
        enc = ProfibusDp.variable_to_bytedata(s, val, CmdArgType.DevLong)
        dec = ProfibusDp.bytedata_to_variable(s, enc, CmdArgType.DevLong)
        assert_equal(f"DevLong {val}", dec, val)

    # DevShort
    for val in [0, 1, -1, 32767, -32768]:
        enc = ProfibusDp.variable_to_bytedata(s, val, CmdArgType.DevShort)
        dec = ProfibusDp.bytedata_to_variable(s, enc, CmdArgType.DevShort)
        assert_equal(f"DevShort {val}", dec, val)

    # DevLong64
    for val in [0, 1, -1, 9223372036854775807, -9223372036854775808]:
        enc = ProfibusDp.variable_to_bytedata(s, val, CmdArgType.DevLong64)
        dec = ProfibusDp.bytedata_to_variable(s, enc, CmdArgType.DevLong64)
        assert_equal(f"DevLong64 {val}", dec, val)

    # DevBoolean (single byte)
    enc = ProfibusDp.variable_to_bytedata(s, True, CmdArgType.DevBoolean)
    dec = ProfibusDp.bytedata_to_variable(s, enc, CmdArgType.DevBoolean)
    assert_true("DevBoolean True round-trip", dec)

    enc = ProfibusDp.variable_to_bytedata(s, False, CmdArgType.DevBoolean)
    dec = ProfibusDp.bytedata_to_variable(s, enc, CmdArgType.DevBoolean)
    assert_false("DevBoolean False round-trip", dec)

    # DevString
    for val in ["Hello", "Profibus", "Test123", ""]:
        enc = ProfibusDp.variable_to_bytedata(s, val, CmdArgType.DevString, suboffset=20)
        dec = ProfibusDp.bytedata_to_variable(s, enc, CmdArgType.DevString)
        assert_equal(f"DevString '{val}'", dec, val)

    # DevString too long raises
    assert_raises("DevString too long", ProfibusDp.variable_to_bytedata,
                  s, "this is too long", CmdArgType.DevString, 5)


def test_output_scalar(s):
    print("\n-- output buffer: scalar types --")

    register_attr(s, "out_float",  "8.o.0",  CmdArgType.DevFloat)
    register_attr(s, "out_double", "8.o.4",  CmdArgType.DevDouble)
    register_attr(s, "out_long",   "8.o.12", CmdArgType.DevLong)
    register_attr(s, "out_short",  "8.o.16", CmdArgType.DevShort)
    register_attr(s, "out_long64", "8.o.18", CmdArgType.DevLong64)

    for val in [0.0, 3.14, -1.5, 1e6]:
        write_attr(s, "out_float", val)
        assert_equal(f"output float {val}", read_attr(s, "out_float"), val, tolerance=max(abs(val) * 1e-6, 1e-7))

    for val in [0.0, 2.718281828459045, -1e100]:
        write_attr(s, "out_double", val)
        assert_equal(f"output double {val}", read_attr(s, "out_double"), val, tolerance=max(abs(val) * 1e-12, 1e-300))

    for val in [0, 1, -1, 2147483647, -2147483648]:
        write_attr(s, "out_long", val)
        assert_equal(f"output long {val}", read_attr(s, "out_long"), val)

    for val in [0, 1, -1, 32767, -32768]:
        write_attr(s, "out_short", val)
        assert_equal(f"output short {val}", read_attr(s, "out_short"), val)

    for val in [0, 1, -1, 9223372036854775807]:
        write_attr(s, "out_long64", val)
        assert_equal(f"output long64 {val}", read_attr(s, "out_long64"), val)


def test_output_string(s):
    print("\n-- output buffer: DevString --")
    register_attr(s, "out_str", "8.o.26.20", CmdArgType.DevString)

    for val in ["Hello", "Profibus", "Test123", ""]:
        write_attr(s, "out_str", val)
        assert_equal(f"output string '{val}'", read_attr(s, "out_str"), val)


def test_output_boolean(s):
    print("\n-- output buffer: DevBoolean (single bit) --")
    register_attr(s, "out_bool", "8.o.50.0", CmdArgType.DevBoolean)

    write_attr(s, "out_bool", True)
    assert_true("output bool True", read_attr(s, "out_bool"))

    write_attr(s, "out_bool", False)
    assert_false("output bool False", read_attr(s, "out_bool"))


def test_output_boolean_multibit(s):
    print("\n-- output buffer: DevBoolean multi-bit (same byte) --")
    register_attr(s, "obit0", "8.o.52.0", CmdArgType.DevBoolean)
    register_attr(s, "obit1", "8.o.52.1", CmdArgType.DevBoolean)
    register_attr(s, "obit2", "8.o.52.2", CmdArgType.DevBoolean)
    register_attr(s, "obit3", "8.o.52.3", CmdArgType.DevBoolean)

    # Set all four bits
    for b in ("obit0", "obit1", "obit2", "obit3"):
        write_attr(s, b, True)
    assert_true("multibit bit0 True", read_attr(s, "obit0"))
    assert_true("multibit bit1 True", read_attr(s, "obit1"))
    assert_true("multibit bit2 True", read_attr(s, "obit2"))
    assert_true("multibit bit3 True", read_attr(s, "obit3"))

    # Clear bit1 only -- others must stay set
    write_attr(s, "obit1", False)
    assert_true("multibit bit0 still True", read_attr(s, "obit0"))
    assert_false("multibit bit1 now False", read_attr(s, "obit1"))
    assert_true("multibit bit2 still True", read_attr(s, "obit2"))
    assert_true("multibit bit3 still True", read_attr(s, "obit3"))

    # Clear all
    for b in ("obit0", "obit2", "obit3"):
        write_attr(s, b, False)
    for b in ("obit0", "obit1", "obit2", "obit3"):
        assert_false(f"multibit {b} cleared", read_attr(s, b))

    # Alternating pattern
    write_attr(s, "obit0", True)
    write_attr(s, "obit2", True)
    assert_true("alternating bit0", read_attr(s, "obit0"))
    assert_false("alternating bit1", read_attr(s, "obit1"))
    assert_true("alternating bit2", read_attr(s, "obit2"))
    assert_false("alternating bit3", read_attr(s, "obit3"))

    # All 8 bits in one byte
    bit_attrs = [f"obit_all{i}" for i in range(8)]
    for i, name in enumerate(bit_attrs):
        register_attr(s, name, f"8.o.54.{i}", CmdArgType.DevBoolean)
    for name in bit_attrs:
        write_attr(s, name, True)
    for name in bit_attrs:
        assert_true(f"{name} set", read_attr(s, name))
    assert_equal("all 8 bits = 0xFF", s._slave_output_data[8][54], 0xFF)
    for name in bit_attrs:
        write_attr(s, name, False)
    assert_equal("all 8 bits cleared = 0x00", s._slave_output_data[8][54], 0x00)


def test_input_scalar(s):
    print("\n-- input buffer: scalar types (pre-seeded) --")

    register_attr(s, "in_float",  "8.i.0",  CmdArgType.DevFloat)
    register_attr(s, "in_double", "8.i.4",  CmdArgType.DevDouble)
    register_attr(s, "in_long",   "8.i.12", CmdArgType.DevLong)
    register_attr(s, "in_short",  "8.i.16", CmdArgType.DevShort)

    # Seed input buffer directly (simulates data arriving from slave)
    val_f = 3.14
    s._slave_input_data[8][0:4] = struct.pack(">f", val_f)
    assert_equal("input float 3.14", read_attr(s, "in_float"), val_f, tolerance=1e-5)

    val_d = 2.718281828459045
    s._slave_input_data[8][4:12] = struct.pack(">d", val_d)
    assert_equal("input double e", read_attr(s, "in_double"), val_d, tolerance=1e-12)

    val_l = -42
    s._slave_input_data[8][12:16] = struct.pack(">i", val_l)
    assert_equal("input long -42", read_attr(s, "in_long"), val_l)

    val_s = 1000
    s._slave_input_data[8][16:18] = struct.pack(">h", val_s)
    assert_equal("input short 1000", read_attr(s, "in_short"), val_s)


def test_input_boolean(s):
    print("\n-- input buffer: DevBoolean (pre-seeded) --")
    register_attr(s, "in_bool0", "8.i.20.0", CmdArgType.DevBoolean)
    register_attr(s, "in_bool3", "8.i.20.3", CmdArgType.DevBoolean)
    register_attr(s, "in_bool7", "8.i.20.7", CmdArgType.DevBoolean)

    s._slave_input_data[8][20] = 0b10001001  # bits 0, 3, 7 set

    assert_true("input bool bit0", read_attr(s, "in_bool0"))
    assert_true("input bool bit3", read_attr(s, "in_bool3"))
    assert_true("input bool bit7", read_attr(s, "in_bool7"))

    register_attr(s, "in_bool1", "8.i.20.1", CmdArgType.DevBoolean)
    assert_false("input bool bit1 clear", read_attr(s, "in_bool1"))


def test_input_string(s):
    print("\n-- input buffer: DevString (pre-seeded) --")
    register_attr(s, "in_str", "8.i.22.10", CmdArgType.DevString)

    val = "sensor"
    raw = val.encode("utf-8") + b"\x00" * (10 - len(val))
    s._slave_input_data[8][22:32] = raw
    assert_equal("input string 'sensor'", read_attr(s, "in_str"), val)


# ===========================================================================
#  Spectrum (1D) tests
# ===========================================================================

def test_spectrum_conversion():
    print("\n-- spectrum conversion (round-trip, no buffer) --")
    s = State()

    for var_type, values, tol in [
        (CmdArgType.DevFloat,  [1.5, -2.5, 3.14], 1e-5),
        (CmdArgType.DevDouble, [1.111, 2.222, 3.333], 1e-9),
        (CmdArgType.DevLong,   [100, -200, 300, 0], None),
        (CmdArgType.DevShort,  [10, -20, 30], None),
    ]:
        elem = ProfibusDp.bytes_per_variable_type(s, var_type)
        raw = b"".join(ProfibusDp.variable_to_bytedata(s, v, var_type) for v in values)
        dec = [ProfibusDp.bytedata_to_variable(s, raw[i*elem:(i+1)*elem], var_type)
               for i in range(len(values))]
        assert_list_equal(f"spectrum {var_type} round-trip", dec, values, tolerance=tol)


def test_spectrum_output(s):
    print("\n-- spectrum output buffer --")

    register_attr(s, "sp_float",  "8.o.60", CmdArgType.DevFloat,  AttrDataFormat.SPECTRUM, max_x=4)
    register_attr(s, "sp_double", "8.o.76", CmdArgType.DevDouble, AttrDataFormat.SPECTRUM, max_x=3)
    register_attr(s, "sp_long",   "8.o.100", CmdArgType.DevLong,  AttrDataFormat.SPECTRUM, max_x=4)
    register_attr(s, "sp_short",  "8.o.116", CmdArgType.DevShort, AttrDataFormat.SPECTRUM, max_x=4)

    vals_f = [1.0, -2.5, 3.14, 0.0]
    write_attr(s, "sp_float", vals_f)
    assert_list_equal("spectrum output float", read_attr(s, "sp_float"), vals_f, tolerance=1e-5)

    vals_d = [1.111, -2.222, 3.333]
    write_attr(s, "sp_double", vals_d)
    assert_list_equal("spectrum output double", read_attr(s, "sp_double"), vals_d, tolerance=1e-9)

    vals_l = [0, 1, -1, 2147483647]
    write_attr(s, "sp_long", vals_l)
    assert_list_equal("spectrum output long", read_attr(s, "sp_long"), vals_l)

    vals_s = [0, 1, -1, 32767]
    write_attr(s, "sp_short", vals_s)
    assert_list_equal("spectrum output short", read_attr(s, "sp_short"), vals_s)


def test_spectrum_input(s):
    print("\n-- spectrum input buffer (pre-seeded) --")
    register_attr(s, "sp_in_float", "8.i.30", CmdArgType.DevFloat, AttrDataFormat.SPECTRUM, max_x=3)

    values = [10.0, 20.0, 30.0]
    raw = b"".join(struct.pack(">f", v) for v in values)
    s._slave_input_data[8][30:30 + len(raw)] = raw

    got = read_attr(s, "sp_in_float")
    assert_list_equal("spectrum input float", got, values, tolerance=1e-5)


# ===========================================================================
#  Image (2D) tests
# ===========================================================================

def test_image_output(s):
    print("\n-- image output buffer --")

    register_attr(s, "img_float",  "8.o.124", CmdArgType.DevFloat, AttrDataFormat.IMAGE, max_x=3, max_y=2)
    register_attr(s, "img_double", "8.o.148", CmdArgType.DevDouble, AttrDataFormat.IMAGE, max_x=2, max_y=2)
    register_attr(s, "img_long",   "8.o.180", CmdArgType.DevLong,  AttrDataFormat.IMAGE, max_x=2, max_y=3)

    data_f = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    write_attr(s, "img_float", data_f)
    assert_2d_equal("image output float", read_attr(s, "img_float"), data_f, tolerance=1e-5)

    data_d = [[1.1, 2.2], [3.3, 4.4]]
    write_attr(s, "img_double", data_d)
    assert_2d_equal("image output double", read_attr(s, "img_double"), data_d, tolerance=1e-9)

    data_l = [[10, 20], [30, 40], [50, 60]]
    write_attr(s, "img_long", data_l)
    assert_2d_equal("image output long", read_attr(s, "img_long"), data_l)


def test_image_input(s):
    print("\n-- image input buffer (pre-seeded) --")
    register_attr(s, "img_in_float", "8.i.50", CmdArgType.DevFloat, AttrDataFormat.IMAGE, max_x=2, max_y=2)

    data = [[1.5, 2.5], [3.5, 4.5]]
    raw = b"".join(struct.pack(">f", v) for row in data for v in row)
    s._slave_input_data[8][50:50 + len(raw)] = raw

    assert_2d_equal("image input float", read_attr(s, "img_in_float"), data, tolerance=1e-5)


# ===========================================================================
#  Edge cases
# ===========================================================================

def test_edge_cases(s):
    print("\n-- edge cases --")
    import math

    register_attr(s, "edge_float",  "8.o.200", CmdArgType.DevFloat)
    register_attr(s, "edge_double", "8.o.204", CmdArgType.DevDouble)
    register_attr(s, "edge_long",   "8.o.212", CmdArgType.DevLong)
    register_attr(s, "edge_str",    "8.o.216.10", CmdArgType.DevString)

    write_attr(s, "edge_float", float("inf"))
    assert_true("float +inf", math.isinf(read_attr(s, "edge_float")) and read_attr(s, "edge_float") > 0)

    write_attr(s, "edge_float", float("-inf"))
    assert_true("float -inf", math.isinf(read_attr(s, "edge_float")) and read_attr(s, "edge_float") < 0)

    write_attr(s, "edge_float", float("nan"))
    assert_true("float nan", math.isnan(read_attr(s, "edge_float")))

    write_attr(s, "edge_double", 0.0)
    assert_equal("double zero", read_attr(s, "edge_double"), 0.0)

    write_attr(s, "edge_long", 0)
    assert_equal("long zero", read_attr(s, "edge_long"), 0)

    write_attr(s, "edge_str", "123456789")  # 9 chars, fits in 10 bytes
    assert_equal("string near-max", read_attr(s, "edge_str"), "123456789")


# ===========================================================================
#  Main
# ===========================================================================

def main():
    global passed, failed

    print("\n== Pure conversion tests (no buffer/connection needed) ==")
    test_parse_register()
    test_byte_conversions()
    test_spectrum_conversion()

    print("\n== Buffer I/O tests ==")
    s = State()
    # Slave 8: 64 bytes input, 256 bytes output (covers all offsets used below)
    s.setup_slave(8, 64, 256)

    test_output_scalar(s)
    test_output_string(s)
    test_output_boolean(s)
    test_output_boolean_multibit(s)
    test_input_scalar(s)
    test_input_boolean(s)
    test_input_string(s)
    test_spectrum_output(s)
    test_spectrum_input(s)
    test_image_output(s)
    test_image_input(s)
    test_edge_cases(s)

    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print("\n  Failures:")
        for e in errors:
            print(f"    {e}")
    print(f"{'=' * 50}")

    sys.exit(0 if failed == 0 else 1)


# ===========================================================================
#  Simulator tests — CpPhyDummySlave (pyprofibus in-process slave)
# ===========================================================================
#
# The dummy slave echoes back every output byte XOR'd with 0xFF.
# To receive a specific value V in the input buffer, write encode(V) ^ 0xFF
# to the output buffer, run a cycle, then read back V via read_attr.
#
# Pyprofibus naming (from the slave's perspective, opposite to our State):
#   input_size  = bytes slave RECEIVES from master  (= our _slave_output_data)
#   output_size = bytes slave SENDS    to master    (= our _slave_input_data)
#
# Configuration: 2 "dummy output module" + 1 "dummy input module"
#   -> input_size=2, output_size=2  (2 bytes each direction)
#
# ===========================================================================

# Slave address and I/O sizes for the simulator slave
SIM_ADDR        = 8
SIM_SEND_SIZE   = 2   # bytes master sends to slave (pyprofibus input_size)
SIM_RECV_SIZE   = 2   # bytes master receives from slave (pyprofibus output_size)


def _write_sim_conf(gsd_path):
    """Write a pyprofibus conf file to a temp file and return its path."""
    content = (
        "[PROFIBUS]\n"
        "debug=0\n"
        "\n"
        "[PHY]\n"
        "type=dummy_slave\n"
        "baud=19200\n"
        "\n"
        "[FDL]\n"
        "\n"
        "[DP]\n"
        "master_class=1\n"
        "master_addr=2\n"
        "\n"
        f"[SLAVE_0]\n"
        f"name=test_slave\n"
        f"addr={SIM_ADDR}\n"
        f"gsd={gsd_path}\n"
        "sync_mode=0\n"
        "freeze_mode=0\n"
        "group_mask=1\n"
        "watchdog_ms=0\n"
        "module_0=dummy output module\n"
        "module_1=dummy output module\n"
        "module_2=dummy input module\n"
        f"input_size={SIM_SEND_SIZE}\n"
        f"output_size={SIM_RECV_SIZE}\n"
        "diag_period=0\n"
    )
    fd, path = tempfile.mkstemp(suffix=".conf")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def setup_simulator_state():
    """
    Create a State backed by a real DPM1 master using CpPhyDummySlave.
    Returns the State on success, or None if pyprofibus is unavailable.
    """
    try:
        from pyprofibus import PbConf
    except ImportError:
        return None

    gsd_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "dummy_modular.gsd")
    )
    if not os.path.isfile(gsd_path):
        print(f"  SKIP  simulator tests: GSD file not found at {gsd_path}")
        return None

    conf_path = _write_sim_conf(gsd_path)
    try:
        pbConf = PbConf.fromFile(conf_path)
        master = pbConf.makeDPM()

        s = State()
        for slaveConf in pbConf.slaveConfs:
            slaveDesc = slaveConf.makeDpSlaveDesc()
            master.addSlave(slaveDesc)
            # pyprofibus naming (slave perspective) vs our naming (master perspective):
            #   slaveConf.outputSize = slave output = master input  -> _slave_input_data
            #   slaveConf.inputSize  = slave input  = master output -> _slave_output_data
            s._slave_descs[slaveConf.addr] = slaveDesc
            s._slave_input_data[slaveConf.addr]  = bytearray(slaveConf.outputSize)
            s._slave_output_data[slaveConf.addr] = bytearray(slaveConf.inputSize)

        master.initialize()
        s._master = master
    finally:
        os.unlink(conf_path)

    return s


def _sim_step(s, max_tries=50):
    """
    Pump master.run() until a DataExchange cycle completes and fresh input data
    is available.  setMasterOutData is called before each run() attempt so the
    current output is always used on the next DataExchange (mirrors the
    pyprofibus example loop pattern).
    """
    slaveDesc = s._slave_descs[SIM_ADDR]
    for _ in range(max_tries):
        slaveDesc.setMasterOutData(bytes(s._slave_output_data[SIM_ADDR]))
        handled = s._master.run()
        if handled is not None:
            data = handled.getMasterInData()
            if data is not None:
                size = len(s._slave_input_data[SIM_ADDR])
                s._slave_input_data[SIM_ADDR][:len(data)] = data[:size]
                return True
    return False


def reach_data_exchange(s, max_cycles=200):
    """Pump the state machine until DataExchange is reached. Returns True on success."""
    for _ in range(max_cycles):
        if _sim_step(s):
            return True
    return False


def test_simulator_state_machine(s):
    print("\n-- simulator: state machine reaches DataExchange --")
    assert_true("DataExchange reached", reach_data_exchange(s))


def test_simulator_xor_echo(s):
    print("\n-- simulator: XOR echo (output ^ 0xFF = input) --")

    # All zeros -> expect all 0xFF in input
    s._slave_output_data[SIM_ADDR][:] = bytearray(SIM_SEND_SIZE)
    _sim_step(s)
    buf = s._slave_input_data[SIM_ADDR]
    assert_equal("xor echo: 0x00 -> 0xFF byte 0", buf[0], 0xFF)
    assert_equal("xor echo: 0x00 -> 0xFF byte 1", buf[1], 0xFF)

    # All 0xFF -> expect all 0x00 in input
    s._slave_output_data[SIM_ADDR][:] = bytearray([0xFF] * SIM_SEND_SIZE)
    _sim_step(s)
    assert_equal("xor echo: 0xFF -> 0x00 byte 0", buf[0], 0x00)
    assert_equal("xor echo: 0xFF -> 0x00 byte 1", buf[1], 0x00)

    # Known pattern
    s._slave_output_data[SIM_ADDR][0] = 0xA5
    s._slave_output_data[SIM_ADDR][1] = 0x3C
    _sim_step(s)
    assert_equal("xor echo: 0xA5 -> 0x5A", buf[0], 0x5A)
    assert_equal("xor echo: 0x3C -> 0xC3", buf[1], 0xC3)


def test_simulator_devshort_roundtrip(s):
    print("\n-- simulator: DevShort round-trip via XOR inversion --")
    register_attr(s, "sim_short", f"{SIM_ADDR}.i.0", CmdArgType.DevShort)

    for val in [0, 1, -1, 32767, -32768, 1234]:
        encoded = struct.pack(">h", val)
        # Write inverted bytes to output so slave echoes back the original
        s._slave_output_data[SIM_ADDR][0:2] = bytes(b ^ 0xFF for b in encoded)
        _sim_step(s)
        got = read_attr(s, "sim_short")
        assert_equal(f"sim DevShort {val}", got, val)


def test_simulator_devboolean_roundtrip(s):
    print("\n-- simulator: DevBoolean round-trip via XOR inversion --")
    register_attr(s, "sim_bool0", f"{SIM_ADDR}.i.0.0", CmdArgType.DevBoolean)
    register_attr(s, "sim_bool1", f"{SIM_ADDR}.i.0.1", CmdArgType.DevBoolean)
    register_attr(s, "sim_bool7", f"{SIM_ADDR}.i.1.7", CmdArgType.DevBoolean)

    # To get bit 0 = True in byte 0: need input byte 0 = 0x01 -> output byte 0 = 0xFE
    s._slave_output_data[SIM_ADDR][0] = 0xFE
    s._slave_output_data[SIM_ADDR][1] = 0xFF
    _sim_step(s)
    assert_true("sim bool bit0 True", read_attr(s, "sim_bool0"))
    assert_false("sim bool bit1 False (byte 0x01, bit 1 = 0)", read_attr(s, "sim_bool1"))

    # To get bit 0 = False: output byte 0 = 0xFF -> input byte 0 = 0x00
    s._slave_output_data[SIM_ADDR][0] = 0xFF
    _sim_step(s)
    assert_false("sim bool bit0 False", read_attr(s, "sim_bool0"))

    # To get byte 1 bit 7 = True: need input byte 1 = 0x80 -> output byte 1 = 0x7F
    s._slave_output_data[SIM_ADDR][1] = 0x7F
    _sim_step(s)
    assert_true("sim bool byte1 bit7 True", read_attr(s, "sim_bool7"))

    # Verify bit isolation: set bit 0 while bit 7 stays True
    s._slave_output_data[SIM_ADDR][0] = 0xFE   # bit 0 -> True in input
    s._slave_output_data[SIM_ADDR][1] = 0x7F   # bit 7 -> True in input
    _sim_step(s)
    assert_true("sim bool isolation: bit0 True", read_attr(s, "sim_bool0"))
    assert_true("sim bool isolation: bit7 True", read_attr(s, "sim_bool7"))


def run_simulator_tests():
    print("\n== Simulator tests (CpPhyDummySlave) ==")
    s = setup_simulator_state()
    if s is None:
        print("  SKIP  pyprofibus not installed — simulator tests skipped")
        return

    try:
        test_simulator_state_machine(s)
        test_simulator_xor_echo(s)
        test_simulator_devshort_roundtrip(s)
        test_simulator_devboolean_roundtrip(s)
    except Exception:
        traceback.print_exc()
        global failed
        failed += 1


def main():
    global passed, failed

    print("\n== Pure conversion tests (no buffer/connection needed) ==")
    test_parse_register()
    test_byte_conversions()
    test_spectrum_conversion()

    print("\n== Buffer I/O tests ==")
    s = State()
    # Slave 8: 64 bytes input, 256 bytes output (covers all offsets used below)
    s.setup_slave(8, 64, 256)

    test_output_scalar(s)
    test_output_string(s)
    test_output_boolean(s)
    test_output_boolean_multibit(s)
    test_input_scalar(s)
    test_input_boolean(s)
    test_input_string(s)
    test_spectrum_output(s)
    test_spectrum_input(s)
    test_image_output(s)
    test_image_input(s)
    test_edge_cases(s)

    run_simulator_tests()

    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print("\n  Failures:")
        for e in errors:
            print(f"    {e}")
    print(f"{'=' * 50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
