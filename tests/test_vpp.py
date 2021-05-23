import unittest
import anytree
import pathlib
from tpp.vpp.core import (
    VerilogASTCompiler,
    VerilogPreprocessor,
    get_verible_verilog_parser,
)
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


class TestVerilogModule(unittest.TestCase):
    def test_ModuleParser(self):
        string = """
        module test1 #(parameter [31:0] a = '1)();
        endmodule
        """
        filepath = str(test_files_dir / "module_parser_test.sv")
        parser = get_verible_verilog_parser()
        module = VerilogModule.from_file(filepath, modulename="test")
