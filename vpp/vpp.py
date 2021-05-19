#!/usr/bin/env python3

from .verible_verilog_syntax import (
    BranchNode,
    RootNode,
    SyntaxData,
    TokenNode,
    LeafNode,
    VeribleVerilogSyntax,
)

from copy import deepcopy
from fnmatch import fnmatch
import os
import sys
import re
import anytree
import math
import importlib
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Generator, Mapping, Optional, Tuple, Union, Dict

INDENT = 2

IdentifierTag = {"tag": ["SymbolIdentifier", "EscapedIdentifier"]}
verilog_number = re.compile(r"(\+-)?(\d)*'([hbodHBOD])?([0-9a-fA-F]+)")
remap_syntax = {
    "===": "==",
    "!==": "!=",
    "&&": "and",
    "||": "or",
}
remap_sysetmTF = {
    "$clog2": "clog2",
}

UNCONNECTED = 0
INTERFACE = 1


def clog2(value):
    value = float(value)
    return math.ceil(math.log2(value))


def eval_expr(expr):
    if not expr:
        return

    expr = "".join(map(str, expr))
    value = eval(expr)
    if isinstance(value, float):
        return math.ceil(value)
    return value


def rename(string: str, repl: Union[dict, Callable]):
    if callable(repl):
        new = repl(string)
    elif repl and isinstance(repl, dict):
        new = repl.get(string, string)
    else:
        new = string
    return new


class DataType:
    def __init__(self, data_type, dimensions: list, default_width: int = None):
        if not isinstance(dimensions, list):
            raise TypeError("expecting a list type")

        self.data_type = data_type or ""
        if default_width is not None:
            self.dimensions = dimensions or [default_width - 1, 0]
        else:
            self.dimensions = dimensions

        self.default_width = default_width

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        return self.data_type != "" or not self.is_unsize

    def __str__(self):
        if not isinstance(self.data_type, str):
            data_type = " ".join(self.data_type)
        else:
            data_type = self.data_type
        if self.size < 2:
            return data_type
        if data_type:
            data_type += " "
        return f"{data_type}[{self.msb}:{self.lsb}]"

    def to_declaration(self, user_data_type):
        if isinstance(self.data_type, tuple):
            data_type = " ".join(self.data_type)
        else:
            data_type = self.data_type or user_data_type
        if self.size < 2:
            return str(data_type)
        else:
            return f"{data_type} [{self.msb}:{self.lsb}]"

    def __len__(self):
        return self.size

    def __format__(self, format_spec):
        return format(self.__str__(), format_spec)

    @property
    def msb(self):
        if self.dimensions:
            return self.dimensions[0]

        return None

    @property
    def lsb(self):
        if len(self.dimensions) == 1:
            return self.dimensions[0]

        elif len(self.dimensions) == 2:
            return self.dimensions[1]

        return None

    @property
    def size(self):
        return abs(self.msb - self.lsb) + 1 if self.dimensions else 1

    @property
    def width(self):
        return self.size

    @property
    def is_unsize(self):
        return len(self.dimensions) == 0


@dataclass
class VerilogModulePort:
    name: str
    direction: str = "inout"
    wire_name: str = None
    data_type: DataType = None
    is_unconnected: bool = True

    def __post_init__(self):
        self.data_type = DataType("", []) if self.data_type is None else self.data_type
        # self.wire_name = self.name if self.wire_name is None else self.wire_name

    def __str__(self):
        data_type = str(self.data_type)
        if data_type:
            return f"{self.direction} {data_type} {self.name}"
        else:
            return f"{self.direction} {self.name}"

    def __repr__(self):
        return self.__str__()

    def conn(self, wire_name):
        self.is_unconnected = False
        self.wire_name = wire_name

    def unconn(self):
        self.is_unconnected = True
        self.wire_name = None

    def get_wire_name(self, default: str = None) -> str:
        if self.wire_name is None:
            return default if default is not None else self.name

        return self.wire_name

    def to_declaration(self, type: str, align: int = 0):
        data_type = str(self.data_type)

        if data_type or align:
            return (
                f"{self.data_type.to_declaration(type):<{align}} {self.get_wire_name()}"
            )
        else:
            return f"{self.data_type.to_declaration(type)} {self.get_wire_name()}"

    def to_wire(self, align: int = 0):
        return self.to_declaration("wire", align)

    def to_logic(self, align: int = 0):
        return self.to_declaration("logic", align)


class VerilogASTCompiler:
    def __init__(self):
        self.defines = {}
        self.stack = []
        self.stack_lvl = -1

    @staticmethod
    def parse_number(iteral, dimension_width=None):
        """
        iteral: verilog integer iteral
        dimension_width: the data_type width of a variable
        for exapmle:
            parameter [2:0] DATA = 3'h1;
            the [2:0] is the data_type width(2 - 0 + 1 = 3)
            the 3'h1 is the iteral
            the DATA is the variable
        """
        if iteral.isdigit():
            return int(iteral)

        match = verilog_number.match(iteral)
        if not match:
            raise ValueError(iteral)
        value = match.group(4).replace("_", "")
        size = int(match.group(2)) if match.group(2) else 32
        fmt = match.group(3) or ""  # format
        fmt = fmt.lower()
        sign = match.group(1) or ""

        # if fmt != "":
        #     if dimension_width is not None and size and dimension_width != size:
        #         """
        #         here means user passed an assigment like:
        #             parameter [11:0] DATA = 11'h02;
        #             The size = 11,
        #             The dimension_width = 11 - 0 + 1 = 12 ([11:0])
        #             11 != 12
        #         """
        #         raise ValueError(
        #             "The data_type range is not euqal than the size of literal"
        #         )

        if sign == "-":
            value = sign + value

        if fmt == "h":
            value = int(value, 16)
        elif fmt == "b":
            value = int(value, 2)
        elif fmt == "o":
            value = int(value, 8)
        elif fmt == "d":
            value = int(value)
        elif fmt == "":
            # reg = '0 or '1 -> all bits of "reg" be zero or one
            # if data_type width is undefine, assume dimension_width = 1 (base on IEEE1800-2017)
            dimension_width = dimension_width or 1
            return int(value * dimension_width, 2)
        else:
            raise NotImplementedError(value)

        binary = bin(value)[2:]
        if len(binary) > size:
            raise ValueError(size)
        return int(binary[:size], 2)

    def visit_pp_node(self, node, callback, tag):
        if not node:
            return
        for child in node.children:
            if child.tag == "kPreprocessorIfdefClause":
                define_id = child.find({"tag": "PP_Identifier"}).text
                match = define_id in self.defines

            elif child.tag == "kPreprocessorIfndefClause":
                define_id = child.find({"tag": "PP_Identifier"}).text
                match = define_id not in self.defines

            elif child.tag == "kPreprocessorElsifClause":
                define_id = child.find({"tag": "PP_Identifier"}).text
                match = define_id in self.defines

            elif child.tag == "kPreprocessorElseClause":
                match = True
            else:
                raise NotImplementedError(child.tag)

            if match:
                # yield from self._walk(child.find(tag))
                yield from callback(child.find(tag))
                break
        return


class VerilogPreprocessor(VerilogASTCompiler):
    """
    Here only evaluates the `ifdef, `ifndef, `elsif, `else, `define and `undef
    Also, this preprocssor only process the description items means that
    any preprocssor statements(or compiler directives) inside the module declaration will be ignored
    """

    def __init__(self, file_path: str, data: SyntaxData):
        self.file_path = file_path
        self.data = data
        self.defines = {}

    def walk_root(self) -> Generator[BranchNode, None, None]:
        yield from self._walk(self.data.tree)

    def _walk(self, tree: Optional[RootNode]) -> Generator[BranchNode, None, None]:
        if not tree:
            return

        for node in tree.children:
            if node.tag == "kPreprocessorDefine":
                define_id = node.find({"tag": "PP_Identifier"}).text
                define_value = node.find({"tag": "PP_define_body"}).text
                self.defines[define_id] = define_value
            elif node.tag == "kPreprocessorUndef":
                # remove a define id
                define_id = node.find({"tag": "PP_Identifier"}).text
                del self.defines[define_id]
            elif node.tag == "kPreprocessorBalancedDescriptionItems":
                yield from self.visit_pp_node(
                    node, self._walk, {"tag": "kDescriptionList"}
                )

            if node.tag == "kModuleDeclaration":
                yield node


class VerilogModuleParser(VerilogASTCompiler):
    def __init__(
        self,
        file_path: str,
        module_tree: BranchNode,
        defines: dict = None,
        params: dict = None,
    ):
        self.file_path = file_path
        self.defines = defines or {}
        self.ports: Dict[str, VerilogModulePort] = {}
        self.module_name = None
        if params and isinstance(params, dict):
            self.params = deepcopy(params)
        else:
            self.params = {}
        self.process_module_tree(module_tree)

    def visit_conditional_expression(self, node):
        # conditional_expression ::=
        # cond_predicate ? { attribute_instance  } expression : expression
        expr = []
        for child in node.children:
            if isinstance(child, TokenNode):
                continue
            node = self.parse_expression(child)
            expr.append(eval_expr(node))

        assert len(expr) == 3, "expecting [cond_predicate, expression, expression]"
        cond_predicate = expr[0]
        assert isinstance(
            cond_predicate, bool
        ), "expecting cond_predicate been evaluated to a boolean type"
        left_expr = expr[1]
        right_expr = expr[2]
        return left_expr if cond_predicate else right_expr

    def visit_dimension_range(self, node):
        if not node:
            return
        for child in node.children:
            if isinstance(child, TokenNode):
                continue
            node = self.parse_expression(child)
            yield eval_expr(node)

    def parse_parameter(self, node):
        name = node.find(IdentifierTag).text
        dimension = node.find({"tag": "kDeclarationDimensions"})
        data_type = DataType("", list(self.parse_expression(dimension)), 32)

        expression = node.find({"tag": "kExpression"})
        value = eval_expr(self.parse_expression(expression, data_type.width))
        return name, value

    def visit_func_call(self, node):
        # global user_defined_functions
        if node.tag == "kSystemTFCall":
            func = node.find({"tag": "SystemTFIdentifier"}).text
            func = remap_sysetmTF.get(func, None)
        if node.tag == "kFunctionCall":
            func = node.find(IdentifierTag).text
            if hasattr(user_defined_functions, func):
                func = f"user_defined_functions.{func}"
            else:
                func = None

        if func is None:
            raise NotImplementedError(
                f"this function '{node.text}' was not implemented yet"
            )

        yield func
        yield from self.parse_expression(node.find({"tag": "kParenGroup"}))

    def parse_expression(self, node, width=None):
        if not node:
            return
        for child in node.children:
            if child.tag == "kNumber":
                yield self.parse_number(child.text, width)
            elif child.tag == "TK_RealTime":
                yield float(child.text)
            elif child.tag == "kReference":
                id_node = child.find({"tag": ["SymbolIdentifier", "MacroIdentifier"]})
                if id_node.tag == "SymbolIdentifier":
                    value = self.params.get(id_node.text, None)
                elif id_node.tag == "MacroIdentifier":
                    value = self.defines.get(id_node.text.replace("`", ""), None)
                else:
                    value = None
                yield value
            # elif child.tag == "kDataTypePrimitive":
            #     yield from self.parse_expression(child)
            elif child.tag == "kConditionExpression":
                yield self.visit_conditional_expression(child)
            elif child.tag == "kDimensionRange":
                yield from self.visit_dimension_range(child)

            elif child.tag in ("kFunctionCall", "kSystemTFCall"):
                yield from self.visit_func_call(child)

            elif isinstance(child, LeafNode):
                yield remap_syntax.get(child.text, child.text)
            else:
                yield from self.parse_expression(child)

    @staticmethod
    def get_module_header_and_name(module) -> Tuple[BranchNode, str]:
        header = module.find({"tag": "kModuleHeader"})
        if not header:
            raise ValueError("ModuleHeader is missing")

        # Find module name
        name = header.find(
            IdentifierTag,
            iter_=anytree.PreOrderIter,
        ).text
        return header, name

    def process_module_tree(self, module_tree):

        # Find module header
        # header = module_tree.find({"tag": "kModuleHeader"})
        # if not header:
        #     raise ValueError("ModuleHeader is missing")
        header, module_name = self.get_module_header_and_name(module_tree)

        self.module_name = module_name

        # Find module port
        port_tag = {"tag": "kPortDeclaration"}
        is_ansi_sytle = header.find(port_tag)

        if is_ansi_sytle:
            paramList = header.find({"tag": "kFormalParameterList"})
            for param in self.visit_parameter_list(paramList):
                name, value = self.parse_parameter(param)
                if name not in self.params:
                    self.params[name] = value

        for port in self.visit_port_declaration_list(header):
            port_id = port.find(IdentifierTag).text
            self.ports[port_id] = VerilogModulePort(name=port_id)

            if port.tag == "kPortDeclaration":
                direction = port.find(lambda x: isinstance(x, TokenNode)).text
                data_type = port.find({"tag": ["kDataType"]})
                interface = data_type.find({"tag": ["kInterfacePortHeader"]})
                if interface:
                    direction = interface.text
                    data_type = ""
                    dimension = []
                else:
                    dimension = data_type.find({"tag": ["kDeclarationDimensions"]})
                    data_type = data_type.find(
                        {"tag": ["kDataTypePrimitive", "kUnqualifiedId"]}
                    )
                    data_type = tuple(self.parse_expression(data_type))
                    dimension = list(self.parse_expression(dimension))

                self.ports[port_id].direction = direction
                self.ports[port_id].data_type = DataType(data_type, dimension)

        if not is_ansi_sytle:
            body = module_tree.find({"tag": "kModuleItemList"})
            if not body:
                raise ValueError("ModuleBody is missing")

            for module_item in self.visit_module_item_list(body):
                if module_item.tag == "kModulePortDeclaration":
                    direction = module_item.children[0].text
                    if module_item.children[1].tag == "kUnqualifiedId":
                        data_type = (module_item.children[1].text,)  # make it a tuple
                    else:
                        data_type = ""
                    dimension = module_item.find({"tag": ["kDeclarationDimensions"]})
                    name = module_item.find(
                        {
                            "tag": [
                                "kIdentifierList",
                                "kIdentifierUnpackedDimensionsList",
                            ]
                        }
                    ).text
                    self.ports[name].data_type = DataType(
                        data_type, list(self.parse_expression(dimension))
                    )
                    self.ports[name].direction = direction

                elif module_item.tag == "kParamDeclaration":
                    name, value = self.parse_parameter(module_item)
                    if name not in self.params:
                        self.params[name] = value

    def visit_port_declaration_list(self, moduleHeader):
        PortDeclarationListTag = {"tag": "kPortDeclarationList"}
        portlist = moduleHeader.find(PortDeclarationListTag)

        for port in portlist.children:
            if port.tag == "kPreprocessorBalancedPortDeclarations":
                yield from self.visit_pp_node(
                    port,
                    self.visit_port_declaration_list,
                    tag=PortDeclarationListTag,
                )
            elif port.tag in ("kPort", "kPortDeclaration"):
                yield port

    def visit_module_item_list(self, moduleBody):
        for child in moduleBody.children:
            if child.tag == "kParamDeclaration":
                yield child
            elif child.tag == "kModulePortDeclaration":
                yield child
            elif child.tag == "kPreprocessorBalancedModuleItems":
                yield from self.visit_pp_node(
                    child,
                    self.visit_module_item_list,
                    tag={"tag": "kModuleItemList"},
                )

    def visit_parameter_list(self, parameterList):
        if not parameterList:
            return
        for child in parameterList.children:
            if child.tag == "kParamDeclaration":
                yield child
            elif child.tag in ("kPreprocessorIfdefClause", "kPreprocessorIfndefClause"):
                raise NotImplementedError(
                    "Currently not support preprocessor directive(ifdef and ifndef) on parameter declaration"
                )


class VerilogModule(VerilogModuleParser):
    def __init__(
        self,
        file_path: str,
        module_tree: BranchNode,
        defines: dict = None,
        params: dict = None,
    ):
        super().__init__(file_path, module_tree, defines, params)
        self.instance_counts = 0
        self.io_conn = {}
        self.io_suffix = ""
        self.io_prefix = ""
        self.io_mode = INTERFACE
        # self.user_params = user_params or {}

    def set_io_mode(self, mode: str):
        modes = (INTERFACE, UNCONNECTED)
        if mode not in modes:
            raise ValueError(f"Only support following modes: {modes}")
        self.io_mode = mode

    def set_io_conn(self, io_conn: Union[dict, Callable]):
        self.io_conn = io_conn

    def set_io_suffix(self, suffix: str):
        self.io_suffix = suffix

    def set_io_prefix(self, prefix: str):
        self.io_prefix = prefix

    def iter_ports(
        self,
        io_conn: Union[dict, Callable] = None,
        prefix: str = None,
        suffix: str = None,
        ignores: list = None,
        filter: Union[str, Callable] = None,
    ) -> Generator[VerilogModulePort, None, None]:

        io_conn = self.io_conn if io_conn is None else io_conn
        suffix = self.io_suffix if suffix is None else suffix
        prefix = self.io_prefix if prefix is None else prefix

        for port_name in io_conn:
            if port_name not in self.ports:
                raise ValueError(
                    f'No such port name "{port_name}" on the module "{self.module_name}"'
                )

        for port in self.ports.values():
            if filter and isinstance(filter, str):
                if not fnmatch(port.name, filter):
                    continue
            if callable(filter):
                if not filter(port.name):
                    continue

            wire_name = rename(port.name, repl=io_conn)
            if wire_name != port.name:
                port.conn(wire_name)
            else:
                port.unconn()
                if self.io_mode == INTERFACE:
                    port.wire_name = f"{prefix}{port.name}{suffix}"
                else:
                    port.wire_name = f"NC__{port.name}"

            if ignores and port.name in ignores:
                continue
            yield port

    def iter_wires(self, ignores: list = None):
        max_len_dimension = max(
            map(
                lambda x: len(str(x.data_type.to_declaration("wire"))),
                self.iter_ports(),
            )
        )
        for port in self.iter_ports(ignores=ignores):
            if (
                self.io_mode == UNCONNECTED
                and port.direction == "input"
                and port.is_unconnected
            ):
                yield f"{port.to_wire(max_len_dimension)} = {port.data_type.size}'h0;"
            else:
                yield f"{port.to_wire(max_len_dimension)};"

    def render_wires(self, ignores: list = None):
        return "\n".join(self.iter_wires(ignores))

    def render_interface(
        self,
        indent=INDENT,
        io_conn: Union[dict, Callable] = None,
        suffix: str = None,
        prefix: str = None,
        include_connected=False,
        ignores: list = None,
    ) -> str:
        indent = indent * " "
        sep = ",\n"
        max_len_dir = max(map(lambda x: len(x.direction), self.ports.values())) + 2
        max_len_dim = max(map(lambda x: len(str(x.data_type)), self.ports.values())) + 2
        ports = sep.join(
            f"{indent}{port.direction:<{max_len_dir}} "
            f"{port.data_type:<{max_len_dim}} "
            f"{port.get_wire_name()}"
            for port in self.iter_ports(io_conn, prefix, suffix, ignores)
            if (port.is_unconnected and self.io_mode == INTERFACE)
            or (include_connected and not port.is_unconnected)
        )
        return ports

    def render_params(self, indent=INDENT) -> str:
        indent = indent * " "
        sep = ",\n"
        params = sep.join(
            f"{indent}parameter {k} = {v}" for k, v in self.params.items()
        )
        return params

    def render_moduleHeader(
        self,
        indent=INDENT,
        io_conn: Union[dict, Callable] = None,
        prefix: str = None,
        suffix: str = None,
        include_connected=False,
    ) -> str:
        return (
            f"module {self.module_name} #(\n"
            f"{self.render_params(indent)}\n"
            f") (\n"
            f"{self.render_interface(indent, io_conn=io_conn, prefix=prefix, suffix=suffix, include_connected=include_connected)}\n"
            f");\n"
        )

    def render_port_connections(
        self,
        indent: int = INDENT,
        io_conn: dict = None,
        suffix: str = None,
        prefix: str = None,
        ignores: list = None,
    ) -> str:

        _indent = indent * " "
        sep = ",\n"
        max_len_name = max(map(lambda x: len(x.name), self.iter_ports()))
        max_len_wire = max(
            map(
                lambda x: len(x.get_wire_name()),
                self.iter_ports(io_conn, prefix, suffix),
            )
        )

        ports = sep.join(
            f"{_indent}.{port.name:<{ max_len_name +2}}"
            f"({port.get_wire_name():<{ max_len_wire +2}}) /* {port} */"
            for port in self.iter_ports(
                io_conn=io_conn, suffix=suffix, prefix=prefix, ignores=ignores
            )
        )
        return ports

    def render_instance(
        self,
        instnace_name: str = None,
        indent: int = INDENT,
        suffix: str = None,
        prefix: str = None,
        params: dict = None,
        io_conn: Union[dict, Callable] = None,
    ) -> str:
        _indent = indent * " "
        params = params or ""

        if params and isinstance(params, dict):
            params = ",\n".join(f"{_indent}.{k}({v})" for k, v in params.items())
            params = f"#(\n{params}\n) "

        if instnace_name is None:
            instnace_name = f"{self.module_name}_{self.instance_counts}"
            self.instance_counts += 1

        ports = self.render_port_connections(
            indent=indent,
            suffix=suffix,
            prefix=prefix,
            io_conn=io_conn,
        )

        return f"{self.module_name} {params}{instnace_name} (\n" f"{ports}\n" f");\n"

    @classmethod
    def from_file(cls, filename: str, modulename: str = None, params: dict = None):
        filepath = Path(os.path.expandvars(filename)).expanduser().resolve()
        if not filepath.exists():
            raise FileNotFoundError(filepath)
        if modulename is None:
            modulename = filepath.stem

        file_data = verible_verilog_parser(str(filepath))
        pp = VerilogPreprocessor(str(filepath), file_data)
        for module in pp.walk_root():
            _, name = cls.get_module_header_and_name(module)
            if modulename == name:
                module = cls(str(filepath), module, pp.defines, params)
                return module
        else:
            raise ValueError(
                f"No such module name: {modulename} in the file 'f{filepath}'"
            )


def verible_verilog_parser(filename: str) -> Optional[Mapping[str, SyntaxData]]:
    parser_path = (
        Path(__file__).parent.resolve().parent
        / "verible"
        / "current"
        / "bin"
        / "verible-verilog-syntax"
    )
    if not parser_path.exists():
        raise FileNotFoundError(parser_path)
    parser = VeribleVerilogSyntax(executable=str(parser_path))
    return parser.parse_file(filename, {"skip_null": True, "gen_rawtokens": False})


def main():
    # if len(sys.argv) < 3:
    #     print(
    #         f"Usage: {sys.argv[0]} PATH_TO_VERIBLE_VERILOG_SYNTAX "
    #         + "VERILOG_FILE [VERILOG_FILE [...]]"
    #     )
    #     return 1

    global user_defined_functions

    if len(sys.argv) > 2:
        user_defined_functions = sys.argv[2]
        user_defined_functions = importlib.import_module(user_defined_functions)
    else:
        user_defined_functions = None

    # files = sys.argv[1:]

    # for file_path, file_data in data.items():
    #     vpp = VerilogPreprocessor(file_path, file_data)
    #     for module in vpp.walk_root():
    #         module = VerilogModuleParser(file_path, module, vpp.defines)
    #         print(module.render_interface(indent=2))
    #         print(module.render_moduleHeader(indent=2))
    #         print(
    #             module.render_instance(
    #                 "test_top",
    #                 suffix="__test",
    #                 params=dict(ADD=9, xxx=123),
    #                 user_ports=dict(clk_i="clk"),
    #             )
    #         )


# if __name__ == "__main__":
#     sys.exit(main())
