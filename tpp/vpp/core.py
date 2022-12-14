#!/usr/bin/env python3

from .verible_verilog_syntax import (
    BranchNode,
    RootNode,
    SyntaxData,
    TokenNode,
    LeafNode,
    VeribleVerilogSyntax,
)

import re
import anytree
import math
from pathlib import Path
from dataclasses import dataclass
from typing import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    Dict,
    Union,
    List,
)
from types import ModuleType
from inspect import getmembers, isfunction

INDENT = 2
USER_DEFINED_FUNCTIONS = Union[ModuleType, Dict[str, Callable]]
PARAMS = Dict[str, Union[str, int]]

IdentifierTag = {"tag": ["SymbolIdentifier", "EscapedIdentifier"]}
remap_syntax = {
    "===": "==",
    "!==": "!=",
    "&&": "and",
    "||": "or",
}

UNCONNECTED = 0
INTERFACE = 1

remap_sysetmTF = {
    "$clog2": "clog2",
}

_globals = {"clog2": lambda value: math.ceil(math.log2(float(value)))}


def _debug(node):
    for prefix, _, child in anytree.RenderTree(node):
        print(prefix, child.to_formatted_string())


class DataType:
    def __init__(self, data_type: str, dimensions: List[int]):
        if not isinstance(dimensions, list):
            raise TypeError("expecting a list type")

        self.data_type = data_type or ""
        if self.data_type in ("reg", "wire", "logic"):
            self.data_type = ""
        assert isinstance(self.data_type, str), "expect a string type"
        # if default_width is not None:
        #     self.dimensions = dimensions or [default_width - 1, 0]
        # else:
        self.dimensions = dimensions

        self.default_width = 32  # TODO: implement this by data type

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        return self.data_type != "" or not self.is_unsize

    def __str__(self):
        # data_type = self.data_type
        # if self.size < 2:
        #     return data_type
        # if data_type:
        #     data_type += " "
        # return f"{data_type}[{self.msb}:{self.lsb}]"
        return self.to_declaration("")

    def to_declaration(self, user_data_type):
        # if isinstance(self.data_type, tuple):
        #     data_type = " ".join(self.data_type)
        # else:
        # data_type = user_data_type or self.data_type
        data_type = self.data_type or user_data_type
        if self.size < 2:
            return data_type
        else:
            if data_type:
                data_type += " "
            return f"{data_type}[{self.msb}:{self.lsb}]"

    def __len__(self):
        return self.size

    def __format__(self, format_spec):
        return format(self.__str__(), format_spec)

    @property
    def type(self):
        return self.data_type

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


class ASSIGNMENT:
    def __init__(self, wire_name: str, value: Union[int, str], _type: str):
        self.wire_name = wire_name
        self.value = value
        self._type = _type

    def __str__(self):
        return self.wire_name

    def __len__(self):
        return len(self.wire_name)

    def __format__(self, format_spec):
        return format(self.__str__(), format_spec)


@dataclass
class VerilogModulePort:
    name: str
    direction: str = "inout"
    wire: Union[ASSIGNMENT, str, None] = None
    data_type: Optional[DataType] = None
    is_unconnected: bool = True

    def __post_init__(self):
        self.data_type = DataType("", []) if self.data_type is None else self.data_type
        # self.wire = self.name if self.wire is None else self.wire

    def __str__(self):
        data_type = str(self.data_type)
        if data_type:
            return f"{self.direction} {data_type} {self.name}"
        else:
            return f"{self.direction} {self.name}"

    def __repr__(self):
        return self.__str__()

    def conn(self, wire: Union[str, ASSIGNMENT]):
        self.is_unconnected = False
        self.wire = wire

    def unconn(self):
        self.is_unconnected = True
        self.wire = None

    @property
    def size(self):
        return self.data_type.size

    @property
    def type(self):
        return self.data_type.type

    @property
    def msb(self):
        return self.data_type.msb

    @property
    def lsb(self):
        return self.data_type.lsb

    @property
    def is_tied(self):
        return isinstance(self.wire, int) or (
            isinstance(self.wire, ASSIGNMENT) and self.wire._type == "tie"
        )

    @property
    def is_assigned(self):
        return isinstance(self.wire, ASSIGNMENT) and self.wire._type == "assign"

    def get_value(self) -> str:
        if self.is_tied or self.is_assigned:
            value = self.wire.value
            if isinstance(value, int):
                return f"{self.size}'h{value:0X}"
            elif isinstance(value, str):
                if value == "'1":
                    return f"{self.size}'h{(1<<self.size)-1:0X}"
                return value
            else:
                raise TypeError(type(value))
        else:
            raise ValueError(f"no value was specified: {self}")

    def get_wire_name(self, default: str = None) -> str:
        if self.wire is None:
            return default if default is not None else self.name
        if self.is_tied:
            return self.get_value()
        if self.is_assigned:
            return self.wire.wire_name
        return self.wire

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

    def to_reg(self, align: int = 0):
        return self.to_declaration("reg", align)


verilog_number = re.compile(r"(\+-)?(\d*)'([hbodHBOD])?([0-9a-fA-F_]+)")


class VerilogASTCompiler:
    def __init__(self):
        self.defines = {}

    @staticmethod
    def parse_number(iteral: str, dimension_width: int = None):
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
            # reg = '0 or '1 -> all bits of "register" be zero or one
            # if data_type width is undefine, assume dimension_width = 1 (base on IEEE1800-2017)
            if dimension_width == 0:
                raise ValueError("diemension should large than 0")
            dimension_width = dimension_width or 1
            return int(value * dimension_width, 2)
        else:
            raise NotImplementedError(value)

        binary = bin(value)[2:]
        # if len(binary) > size:
        #     raise ValueError(f"Overflow: value bin length large t")
        return int(binary[-size:], 2)

    def visit_pp_node(self, node: BranchNode) -> Generator[BranchNode, None, None]:
        if not node:
            return
        for child in node.children:
            if child.tag in ("kPreprocessorIfdefClause", "kPreprocessorElsifClause"):
                define_id = child.find({"tag": "PP_Identifier"})
                match = define_id.text in self.defines

            elif child.tag == "kPreprocessorIfndefClause":
                define_id = child.find({"tag": "PP_Identifier"})
                match = define_id.text not in self.defines

            # elif child.tag == "kPreprocessorElsifClause":
            #     define_id = child.find({"tag": "PP_Identifier"}).text
            #     match = define_id in self.defines

            elif child.tag == "kPreprocessorElseClause":
                match = True
            # else:
            #     raise NotImplementedError(child.tag)

            if match:
                # yield from self._walk(child.find(tag))
                # yield from callback(child.find(tag))
                yield child
                break
        return


class VerilogPreprocessor(VerilogASTCompiler):
    """
    Here only evaluates the `ifdef, `ifndef, `elsif, `else, `define and `undef and does not
    evaluation the expression or statement.
    Also, this preprocssor only process the description items means that
    any preprocssor statements(or compiler directives) inside the module declaration will be ignored
    """

    def __init__(self, filepath: str, data: SyntaxData, defines: PARAMS = None):
        self.data = data
        self.filepath = filepath
        if self.data.errors:
            raise SyntaxError(
                "\n"
                + "\n".join(
                    f"{filepath}:{error.line}:{error.column}: syntax error, {error.text}"
                    for error in self.data.errors
                )
            )
        self.defines = defines or {}

    def walk_root(self) -> Generator[BranchNode, None, None]:
        yield from self._walk(self.data.tree)

    def _walk(self, tree: Optional[RootNode]) -> Generator[BranchNode, None, None]:
        if not tree:
            return

        for node in tree.children:
            if node.tag == "kPreprocessorDefine":
                define_id = node.find({"tag": "PP_Identifier"}).text
                define_value = node.find({"tag": "PP_define_body"}).text or ""
                while "`" in define_value:
                    old = define_value
                    for id, value in self.defines.items():
                        define_value = define_value.replace(f"`{id}", value)
                    if define_value == old:
                        if "`" in define_value:
                            raise ValueError(
                                f"No such marco indentifier: {define_value}"
                            )
                        break

                self.defines[define_id] = define_value
            elif node.tag == "kPreprocessorUndef":
                # remove a define id
                define_id = node.find({"tag": "PP_Identifier"}).text
                del self.defines[define_id]
            elif node.tag == "kPreprocessorBalancedDescriptionItems":
                # yield from self.visit_pp_node(
                #     node, self._walk, {"tag": "kDescriptionList"}
                # )
                for child in self.visit_pp_node(node):
                    yield from self._walk(child.find({"tag": "kDescriptionList"}))

            if node.tag == "kModuleDeclaration":
                yield node

    @classmethod
    def from_string(cls, string: str, defines: PARAMS = None):
        parser = get_verible_verilog_parser()
        return cls(
            "from_string",
            parser.parse_string(string, {"skip_null": True, "gen_rawtokens": False}),
            defines,
        )

    @classmethod
    def from_file(cls, filepath: str, defines: PARAMS = None):
        parser = get_verible_verilog_parser()
        return cls(
            filepath,
            parser.parse_file(filepath, {"skip_null": True, "gen_rawtokens": False}),
            defines,
        )


class VerilogModuleParser(VerilogASTCompiler):
    def __init__(
        self,
        filepath: str,
        module_tree: BranchNode,
        defines: PARAMS = None,
        params: PARAMS = None,
        user_defined_functions: USER_DEFINED_FUNCTIONS = None,
    ):
        self.data = module_tree
        self.filepath = filepath
        self.defines = defines or {}
        self.ports: Dict[str, VerilogModulePort] = {}
        self.module_name: str = None
        if isinstance(user_defined_functions, ModuleType):
            self.user_defined_functions = dict(
                getmembers(user_defined_functions, isfunction)
            )
        elif isinstance(user_defined_functions, dict):
            self.user_defined_functions = user_defined_functions.copy()
        elif user_defined_functions:
            raise NotImplementedError(
                f"user_defined_functions should be a moulde or dict, got f{type(user_defined_functions)} instead"
            )
        else:
            self.user_defined_functions = {}

        if isinstance(params, dict):
            self.params = params.copy()
        else:
            self.params = {}

        self.process_module_tree(module_tree)

    def eval_expr(self, expr: Iterable):
        if not expr:
            return
        expr = "".join(map(str, expr))
        try:
            value = eval(expr, _globals, self.user_defined_functions)
        except SyntaxError:
            raise SyntaxError(expr or "got empty string")
        if isinstance(value, float):
            return math.ceil(value)
        return value

    def visit_conditional_expression(self, node: BranchNode):
        # conditional_expression ::=
        # cond_predicate ? { attribute_instance  } expression : expression
        expr = []
        for index, child in enumerate(node.children):
            # node.children should be = [expr, ?, expr, :, expr]
            # so only parse child if index is even number,
            if index % 2:  # index is odd number
                continue
            node = self.parse_expression(child)
            expr.append(self.eval_expr(node))

        assert (
            len(expr) == 3
        ), f"expecting [cond_predicate, expression, expression], got {expr}"
        cond_predicate = expr[0]
        assert isinstance(
            cond_predicate, bool
        ), "expecting cond_predicate been evaluated to a boolean type"
        left_expr = expr[1]
        right_expr = expr[2]
        result = left_expr if cond_predicate else right_expr
        if isinstance(result, str):
            return f"'{result}'"
        return result

    def visit_dimension_range(self, node: BranchNode):
        if not node:
            return
        for child in node.children:
            if isinstance(child, TokenNode):
                continue
            node = self.parse_expression(child)
            yield self.eval_expr(node)

    def parse_parameter(self, node: BranchNode):
        name = node.find(IdentifierTag).text
        # _debug(node)
        if node.find({"tag": "kTypeAssignment"}):
            # parameter type ID = value
            value = node.find({"tag": "kDataType"}).text
            data_type = DataType("type", [])
        else:
            dimension = node.find({"tag": "kDeclarationDimensions"})
            _type = node.find({"tag": "kTypeInfo"}).text or ""
            data_type = DataType(_type, list(self.parse_expression(dimension)))

            expression = node.find({"tag": "kExpression"})
            value = self.eval_expr(self.parse_expression(expression, data_type.width))
        return name, value

    def visit_func_call(self, node: BranchNode):
        if node.tag == "kSystemTFCall":
            func = node.find({"tag": "SystemTFIdentifier"}).text
            func = remap_sysetmTF.get(func, None)
        if node.tag == "kFunctionCall":
            func = node.find(IdentifierTag).text
            if func not in self.user_defined_functions:
                func = None
        if func is None:
            raise NotImplementedError(
                f"this function '{node.text}' was not implemented"
            )

        yield func
        yield from self.parse_expression(node.find({"tag": "kParenGroup"}))

    def parse_expression(self, node: BranchNode, width: int = None):
        if not node:
            return
        try:
            # for node in node.noderen:
            if node.tag == "kNumber":
                # a iteral
                yield self.parse_number(node.text, width)
            elif node.tag == "TK_RealTime":
                # a iteral
                yield float(node.text)
            elif node.tag == "TK_StringLiteral":
                yield f"'{node.text}'"
            elif node.tag == "kReference":
                # refernce a parameter or a define
                id_node = node.find({"tag": ["SymbolIdentifier", "MacroIdentifier"]})
                if id_node.tag == "SymbolIdentifier":
                    value = self.params.get(id_node.text, None)
                elif id_node.tag == "MacroIdentifier":
                    value = self.defines.get(id_node.text.replace("`", ""), None)
                else:
                    raise NotImplementedError(id_node.tag)
                if value is None:
                    raise ValueError(f"No such parameter or define: {id_node.text}")
                yield value
            elif node.tag == "kConditionExpression":
                # conditional expression = a ? b : c
                yield self.visit_conditional_expression(node)
            elif node.tag == "kDimensionRange":
                # parse [31:0] or [ADDR-1:0]
                yield from self.visit_dimension_range(node)
            elif node.tag in ("kFunctionCall", "kSystemTFCall"):
                yield from self.visit_func_call(node)
            elif isinstance(node, LeafNode):
                yield remap_syntax.get(node.text, node.text)
            else:
                for child in node.children:
                    yield from self.parse_expression(child, width)
        except:
            raise NotImplementedError(node.text)

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

            # _debug(port)
            if port.tag == "kPortDeclaration":
                direction = port.find(lambda x: isinstance(x, TokenNode)).text
                data_type = port.find({"tag": ["kDataType"]})
                interface = data_type.find({"tag": ["kInterfacePortHeader"]})
                if interface:
                    direction = interface.text
                    data_type_id = ""
                    dimension = []
                else:
                    data_type_id = ""
                    for child in data_type.children:
                        if child.tag == "kDataTypePrimitive":
                            data_type_id = child.text
                        elif child.tag == "kUnqualifiedId":
                            data_type_id = child.text
                            data_type_id = self.params.get(data_type_id, data_type_id)

                    dimension = data_type.find({"tag": ["kDeclarationDimensions"]})
                    dimension = list(self.parse_expression(dimension))

                self.ports[port_id].direction = direction
                self.ports[port_id].data_type = DataType(str(data_type_id), dimension)

        if not is_ansi_sytle:
            body = module_tree.find({"tag": "kModuleItemList"})
            if not body:
                raise ValueError("ModuleBody is missing")

            for module_item in self.visit_module_item_list(body):
                if module_item.tag == "kModulePortDeclaration":
                    direction = module_item.children[0].text
                    _type_node = module_item.children[1]
                    if _type_node.tag == "kUnqualifiedId":
                        # a user-defined type
                        data_type: str = _type_node.text
                        data_type = self.params.get(data_type, data_type)
                    elif isinstance(_type_node, TokenNode):
                        data_type: str = _type_node.text
                        if data_type == "wire":
                            data_type = ""  # here we drop wire type
                    else:
                        data_type: str = ""
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

    def visit_port_declaration_list(
        self, moduleHeader: BranchNode
    ) -> Iterator[BranchNode]:
        PortDeclarationListTag = {"tag": "kPortDeclarationList"}
        portlist = moduleHeader.find(PortDeclarationListTag)
        if not portlist:
            raise ValueError("failed to locate port declaration list")

        for port in portlist.children:
            if port.tag == "kPreprocessorBalancedPortDeclarations":
                for child in self.visit_pp_node(port):
                    yield from self.visit_port_declaration_list(
                        child.find(PortDeclarationListTag)
                    )
            elif port.tag in ("kPort", "kPortDeclaration"):
                yield port

    def visit_module_item_list(self, moduleBody: BranchNode):
        for child in moduleBody.children:
            if child.tag == "kParamDeclaration":
                yield child
            elif child.tag == "kModulePortDeclaration":
                yield child
            elif child.tag == "kPreprocessorBalancedModuleItems":
                for grandchild in self.visit_pp_node(child):
                    yield from self.visit_module_item_list(
                        grandchild.find({"tag": "kModuleItemList"})
                    )

    def visit_parameter_list(self, parameterList: BranchNode):
        if not parameterList:
            return
        for child in parameterList.children:
            if child.tag == "kParamDeclaration":
                yield child
            elif child.tag in ("kPreprocessorIfdefClause", "kPreprocessorIfndefClause"):
                raise NotImplementedError(
                    "Currently not support preprocessor directive(ifdef and ifndef) on parameter declaration"
                )


def get_verible_verilog_parser() -> VeribleVerilogSyntax:
    parser_path = (
        Path(__file__).parent.resolve().parent
        / "verible"
        / "current"
        / "bin"
        / "verible-verilog-syntax"
    )
    if not parser_path.exists():
        raise FileNotFoundError(parser_path)
    return VeribleVerilogSyntax(executable=str(parser_path))