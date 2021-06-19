import unittest
import pathlib
from tpp.vpp.core import VerilogASTCompiler, VerilogPreprocessor, _debug
from tpp.vpp.vpp import VerilogModule

test_files_dir = pathlib.Path(__file__).parent.resolve() / "test_files"


class TestParseNumber(unittest.TestCase):
    def test_parse_number(self):
        parse_number = VerilogASTCompiler.parse_number
        self.assertEqual(parse_number("32'h123"), 0x123)
        self.assertEqual(parse_number("'h123"), 0x123)
        self.assertEqual(parse_number("'b1001"), 0b1001)
        self.assertEqual(parse_number("3'b1_1001"), 0b1)
        self.assertEqual(parse_number("4'd15"), 15)
        self.assertEqual(parse_number("3'D15"), 7)
        self.assertEqual(parse_number("'_1"), 1)
        self.assertEqual(parse_number("'0_"), 0)
        self.assertEqual(parse_number("'1", 1), 0x1)
        self.assertEqual(parse_number("'1", 2), 0x3)
        self.assertEqual(parse_number("'1", 4), 0xF)
        self.assertEqual(parse_number("'1", 8), 0xFF)
        self.assertEqual(parse_number("'1", 16), 0xFFFF)
        self.assertEqual(parse_number("'1", 32), 0xFFFFFFFF)
        self.assertEqual(parse_number("'1", 33), 0x1FFFFFFFF)
        self.assertEqual(parse_number("'1", 64), 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(parse_number("32'hFFFF_FF_FF"), 0xFFFFFFFF)
        self.assertEqual(parse_number("31'HFFFF_FFF_F"), 0x7FFFFFFF)
        self.assertEqual(parse_number("30'hFFFF_FF00__"), 0x3FFFFF00)
        self.assertEqual(parse_number("123"), 123)
        with self.assertRaises(ValueError):
            self.assertEqual(parse_number("'1", 0), 0x0)
        with self.assertRaises(ValueError):
            self.assertEqual(parse_number("_", 1), 0x1)
        with self.assertRaises(ValueError):
            parse_number("'b23")
        with self.assertRaises(ValueError):
            parse_number("'c23")


class TestVerilogPreprocessor(unittest.TestCase):
    def test_verilog_preprocessor(self):
        vpp = VerilogPreprocessor.from_file(str(test_files_dir / "defines.svh"))
        iter_modules = vpp.walk_root()
        next(iter_modules)
        self.assertIn("test_empty_value", vpp.defines)
        self.assertIn("test_value", vpp.defines)
        self.assertIn("test_string", vpp.defines)
        self.assertIn("test_get_value", vpp.defines)
        self.assertIn("else", vpp.defines)
        self.assertIn("elsif_succeeded", vpp.defines)
        self.assertNotIn("test_not_exists", vpp.defines)
        self.assertNotIn("not_in", vpp.defines)
        self.assertNotIn("elsif", vpp.defines)
        self.assertNotIn("YES", vpp.defines)
        self.assertEqual(vpp.defines.get("else"), '"success"')
        next(iter_modules)
        self.assertNotIn("test_value", vpp.defines)
        self.assertNotIn("test_empty_value", vpp.defines)
        self.assertIn("YES", vpp.defines)
        self.assertEqual(vpp.defines.get("else"), "0")

        with self.assertRaises(ValueError):
            next(iter_modules)
        with self.assertRaises(StopIteration):
            next(iter_modules)

        vpp = VerilogPreprocessor.from_file(
            str(test_files_dir / "defines_from_args.svh"), defines={"YES": ""}
        )
        iter_modules = vpp.walk_root()
        with self.assertRaises(StopIteration):
            next(iter_modules)
        self.assertIn("YES_exists", vpp.defines)
        self.assertIn("YES", vpp.defines)
        self.assertIn("COOL_not_exists", vpp.defines)
        self.assertNotIn("COOL_exists", vpp.defines)
        self.assertNotIn("YES_not_exists", vpp.defines)


class TestVerilogModule__ANSIHeader(unittest.TestCase):
    def setUp(self) -> None:
        filepath = str(test_files_dir / "module_ansi.sv")
        self.module = VerilogModule.from_file(filepath, modulename="test")

    def test_params(self):
        self.assertEqual(self.module.params["ADDR"], 32)
        self.assertEqual(self.module.params["SIZE"], 0xFFFF)
        self.assertEqual(self.module.params["WORD"], 3 << 3)
        self.assertEqual(self.module.params["STAGE"], '"ASIC"')
        self.assertEqual(self.module.params["cond_s0"], 1)
        self.assertEqual(self.module.params["cond_s1"], 0)
        self.assertEqual(self.module.params["cond0"], '"YES"')
        self.assertEqual(self.module.params["cond1"], 1)
        self.assertEqual(self.module.params["PTR"], 5)
        self.assertEqual(self.module.params["store_t"], "logic")
        # self.assertEqual(self.module.params["user_t"], "logic [31:0]")

    def test_iter_ports(self):
        self.assertEqual(len(list(self.module.ports)), 15)
        self.assertEqual(len(list(self.module.iter_ports(patterns="w*"))), 4)
        self.assertEqual(
            len(list(self.module.iter_ports(patterns=r"w.*", regxp=True))), 4
        )
        self.assertEqual(len(list(self.module.iter_ports(patterns="*_if"))), 2)
        self.assertEqual(
            len(list(self.module.iter_ports(patterns=r".*_if$", regxp=True))), 2
        )
        self.assertEqual(len(list(self.module.iter_ports(patterns=["w*", "r*"]))), 8)

        self.assertEqual(
            len(list(self.module.iter_ports(ignores=[r"w.*"], regxp=True))), 11
        )
        self.assertEqual(len(list(self.module.iter_ports(ignores=[r"w*"]))), 11)

    def test_ports(self):
        items = [
            # name, type, size, msb, lsb
            ["rstn_i", "", 2, 1, 0],
            ["clk_i", "", 1, None, None],
            ["addr", "", 32, 31, 0],
            ["pop", "", 1, None, None],
            ["push", "", 1, None, None],
            ["wstrb", "", 8, 7, 0],
            ["wuser", "logic [31:0]", 1, None, None],
            ["ruser", "", 32, 31, 0],
            ["regs_if", "", 1000, 1024, 25],
            ["bus_if", "", 7, 15, 9],
        ]
        for item in items:
            port = self.module.ports.get(item[0])
            self.assertEqual(port.type, item[1], str(port))
            self.assertEqual(port.size, item[2], str(port))
            self.assertEqual(port.msb, item[3], str(port))
            self.assertEqual(port.lsb, item[4], str(port))

        self.assertIsNone(self.module.ports.get("fpga_clk"))


# @unittest.skip
class TestVerilogModule__ANSIHeaderWithDefines(unittest.TestCase):
    def setUp(self) -> None:
        filepath = str(test_files_dir / "module_ansi.sv")
        self.module = VerilogModule.from_file(
            filepath,
            modulename="test",
            defines={"tri_clk": "", "INTERFACE": "", "FPGA": ""},  # defines
        )

    def test_ports(self):
        items = [
            # name, type, size, msb, lsb, direction
            ["rstn_i", "", 2, 1, 0, "input"],
            ["clk_i", "", 3, 2, 0, "input"],
            ["addr", "", 32, 31, 0, "input"],
            ["pop", "", 1, None, None, "input"],
            ["push", "", 1, None, None, "input"],
            ["wstrb", "", 8, 7, 0, "input"],
            ["wuser", "logic [31:0]", 1, None, None, "input"],
            ["ruser", "", 32, 31, 0, "output"],
            ["regs_if0", "", 1, None, None, "reg_if.slv"],
            ["bus_if0", "", 1, None, None, "bus_if.mst"],
            ["inout_port", "", 1, 0, 0, "inout"],
        ]
        for item in items:
            port = self.module.ports.get(item[0], None)
            self.assertEqual(port.type, item[1], str(port))
            self.assertEqual(port.size, item[2], str(port))
            self.assertEqual(port.msb, item[3], str(port))
            self.assertEqual(port.lsb, item[4], str(port))
            self.assertEqual(port.direction, item[5], str(port))

        self.assertIsNone(self.module.ports.get("regs_if"))
        self.assertIsNone(self.module.ports.get("bus_if"))
        self.assertIsNotNone(self.module.ports.get("fpga_clk"))


# @unittest.skip
class TestVerilogModule__nonANSIHeader(unittest.TestCase):
    def setUp(self) -> None:
        filepath = str(test_files_dir / "module_non_ansi.sv")
        self.module = VerilogModule.from_file(filepath, modulename="test")

    def test_params(self):
        self.assertEqual(self.module.params["ADDR"], 32)
        self.assertEqual(self.module.params["SIZE"], 0xFFFF)
        self.assertEqual(self.module.params["WORD"], 3 << 3)
        self.assertEqual(self.module.params["STAGE"], '"ASIC"')
        self.assertEqual(self.module.params["cond_s0"], 1)
        self.assertEqual(self.module.params["cond_s1"], 0)
        self.assertEqual(self.module.params["cond0"], '"YES"')
        self.assertEqual(self.module.params["cond1"], 1)
        self.assertEqual(self.module.params["PTR"], 5)
        self.assertEqual(self.module.params["store_t"], "logic")
        # self.assertEqual(self.module.params["user_t"], "logic [31:0]")

    def test_iter_ports(self):
        self.assertEqual(len(list(self.module.ports)), 15)
        self.assertEqual(len(list(self.module.iter_ports(patterns="w*"))), 4)
        self.assertEqual(
            len(list(self.module.iter_ports(patterns=r"w.*", regxp=True))), 4
        )
        self.assertEqual(len(list(self.module.iter_ports(patterns="*_if"))), 2)
        self.assertEqual(
            len(list(self.module.iter_ports(patterns=r".*_if$", regxp=True))), 2
        )
        self.assertEqual(len(list(self.module.iter_ports(patterns=["w*", "r*"]))), 8)

        self.assertEqual(
            len(list(self.module.iter_ports(ignores=[r"w.*"], regxp=True))), 11
        )
        self.assertEqual(len(list(self.module.iter_ports(ignores=[r"w*"]))), 11)

    def test_ports(self):
        items = [
            # name, type, size, msb, lsb
            ["rstn_i", "", 2, 1, 0],
            ["clk_i", "", 1, None, None],
            ["addr", "", 32, 31, 0],
            ["pop", "", 1, None, None],
            ["push", "logic [31:0]", 1, None, None],
            ["wstrb", "", 8, 7, 0],
            ["wuser", "user_t", 1, None, None],
            ["ruser", "", 32, 31, 0],
            ["regs_if", "", 1000, 1024, 25],
            ["bus_if", "", 7, 15, 9],
        ]
        for item in items:
            port = self.module.ports.get(item[0])
            self.assertEqual(port.type, item[1], str(port))
            self.assertEqual(port.size, item[2], str(port))
            self.assertEqual(port.msb, item[3], str(port))
            self.assertEqual(port.lsb, item[4], str(port))

        self.assertIsNone(self.module.ports.get("fpga_clk"))


# @unittest.skip
class TestVerilogModule__nonANSIHeaderWithDefines(unittest.TestCase):
    def setUp(self) -> None:
        filepath = str(test_files_dir / "module_non_ansi.sv")
        self.module = VerilogModule.from_file(
            filepath,
            modulename="test",
            # Not support interface with non ANSI header
            defines={"tri_clk": "", "FPGA": ""},  # defines
        )

    def test_ports(self):
        items = [
            # name, type, size, msb, lsb, direction
            ["rstn_i", "", 2, 1, 0, "input"],
            ["clk_i", "", 3, 2, 0, "input"],
            ["addr", "", 32, 31, 0, "input"],
            ["pop", "", 1, None, None, "input"],
            ["push", "logic [31:0]", 1, None, None, "input"],
            ["wstrb", "", 8, 7, 0, "input"],
            ["wuser", "user_t", 1, None, None, "input"],
            ["ruser", "", 32, 31, 0, "output"],
            ["inout_port", "", 1, 0, 0, "inout"],
            ["regs_if", "", 1000, 1024, 25, "input"],
            ["bus_if", "", 7, 15, 9, "output"],
        ]
        for item in items:
            port = self.module.ports.get(item[0], None)
            self.assertEqual(port.type, item[1], str(port))
            self.assertEqual(port.size, item[2], str(port))
            self.assertEqual(port.msb, item[3], str(port))
            self.assertEqual(port.lsb, item[4], str(port))
            self.assertEqual(port.direction, item[5], str(port))

        self.assertIsNone(self.module.ports.get("regs_if0"))
        self.assertIsNone(self.module.ports.get("bus_if0"))
        self.assertIsNotNone(self.module.ports.get("fpga_clk"))