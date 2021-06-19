#!/usr/bin/env python3

from copy import deepcopy
import os
import re

from pathlib import Path
from fnmatch import fnmatch
from typing import (
    Iterator,
    TypedDict,
    Union,
    Callable,
    List,
    Dict,
    Optional,
)
from .core import (
    VerilogModuleParser,
    VerilogModulePort,
    VerilogPreprocessor,
    get_verible_verilog_parser,
    USER_DEFINED_FUNCTIONS,
    PARAMS,
)
from .verible_verilog_syntax import BranchNode

INDENT: int = 2
UNCONNECTED: int = 0
INTERFACE: int = 1
IO_CONNECT = Union[Dict[str, str], Callable[[str], Optional[str]]]
PATTERN = Union[List[str], str, Callable[[str], bool]]


def connect(string: str, repl: IO_CONNECT) -> Optional[str]:
    new = None
    if callable(repl):
        new = repl(string)
    elif isinstance(repl, dict):
        new = repl.get(string, None)
    return new


class IO(TypedDict, total=False):
    mode: int
    suffix: str
    prefix: str
    connect: IO_CONNECT


class VerilogModule(VerilogModuleParser):
    def __init__(
        self,
        file_path: str,
        module_tree: BranchNode,
        defines: PARAMS = None,
        params: PARAMS = None,
        user_defined_functions: USER_DEFINED_FUNCTIONS = None,
        io: IO = None,
    ):
        self.params_redefined = params.copy() if isinstance(params, dict) else {}
        self.instance_counts = 0
        self.io: IO = dict(mode=INTERFACE, suffix="", prefix="", connect={})
        if isinstance(io, dict):
            self.io.update(deepcopy(io))
        super().__init__(
            file_path, module_tree, defines, params, user_defined_functions
        )

    def set_io_mode(self, mode: int):
        modes = (INTERFACE, UNCONNECTED)
        if mode not in modes:
            raise ValueError(f"Only support following modes: {modes}")
        self.io["mode"] = mode

    def set_io_connect(self, connect: IO_CONNECT):
        self.io["connect"] = connect

    def set_io_suffix(self, suffix: str):
        self.io["suffix"] = suffix

    def set_io_prefix(self, prefix: str):
        self.io["prefix"] = prefix

    def iter_ports(
        self,
        io: IO = None,
        ignores: PATTERN = None,
        patterns: PATTERN = None,
        regxp: bool = False,
    ) -> Iterator[VerilogModulePort]:

        io = {**self.io, **io} if isinstance(io, dict) else self.io
        io_connect = io["connect"]
        io_suffix = io["suffix"]
        io_prefix = io["prefix"]
        io_mode = io["mode"]

        def _match(patterns: PATTERN, string: str) -> bool:
            if isinstance(patterns, str):
                if not regxp:
                    return fnmatch(string, patterns)
                if regxp:
                    return re.match(patterns, string)
            if callable(patterns):
                return patterns(string)
            if isinstance(patterns, list):
                return any(filter(lambda p: _match(p, string), patterns))
            return False

        for port in self.ports.values():
            if ignores and _match(ignores, port.name):
                continue
            if patterns and not _match(patterns, port.name):
                continue
            wire_name = connect(port.name, repl=io_connect)
            if wire_name is not None:
                port.conn(wire_name)
            else:
                port.unconn()
                if io_mode == INTERFACE:
                    port.wire_name = f"{io_prefix}{port.name}{io_suffix}"
                else:
                    port.wire_name = f"NC__{port.name}"

            yield port

    def iter_declarations(self, data_type: str, ignores: List[str] = None):
        max_len_dimension = max(
            map(
                lambda x: len(str(x.data_type.to_declaration(data_type))),
                self.iter_ports(),
            )
        )
        for port in self.iter_ports(ignores=ignores):
            if (
                self.io["mode"] == UNCONNECTED
                and port.direction == "input"
                and port.is_unconnected
            ):
                yield f"{port.to_declaration(data_type, max_len_dimension)} = {port.data_type.size}'h0;"
            else:
                yield f"{port.to_declaration(data_type, max_len_dimension)};"

    def io_to_wires(self, ignores: List[str] = None) -> str:
        return "\n".join(self.iter_declarations("wire", ignores))

    def io_to_logics(self, ignores: List[str] = None) -> str:
        return "\n".join(self.iter_declarations("logic", ignores))

    def render_interface(
        self,
        indent: int = INDENT,
        io: IO = None,
        include_connected=False,
        ignores: List[str] = None,
    ) -> str:
        io_mode = (
            io.get("mode", self.io["mode"]) if isinstance(io, dict) else self.io["mode"]
        )
        indent = indent * " "
        sep = ",\n"
        max_len_dir = max(map(lambda x: len(x.direction), self.ports.values())) + 2
        max_len_dim = max(map(lambda x: len(str(x.data_type)), self.ports.values())) + 2
        ports = sep.join(
            f"{indent}{port.direction:<{max_len_dir}} "
            f"{port.data_type:<{max_len_dim}} "
            f"{port.get_wire_name()}"
            for port in self.iter_ports(io=io, ignores=ignores)
            if (port.is_unconnected and io_mode == INTERFACE)
            or (include_connected and not port.is_unconnected)
        )
        return ports

    def render_params(self, indent: int = INDENT) -> str:
        indent = indent * " "
        sep = ",\n"
        params = sep.join(
            f"{indent}parameter {k} = {v}" for k, v in self.params.items()
        )
        return params

    def render_moduleHeader(
        self,
        indent: int = INDENT,
        io: IO = None,
        include_connected: bool = False,
    ) -> str:
        return (
            f"module {self.module_name} #(\n"
            f"{self.render_params(indent)}\n"
            f") (\n"
            f"{self.render_interface(indent, io=io, include_connected=include_connected)}\n"
            f");\n"
        )

    def render_port_connections(
        self,
        indent: int = INDENT,
        io: IO = None,
        ignores: List[str] = None,
    ) -> str:

        _indent = indent * " "
        sep = ",\n"
        max_len_name = max(map(lambda x: len(x.name), self.iter_ports()))
        max_len_wire = max(
            map(
                lambda x: len(x.get_wire_name()),
                self.iter_ports(io=io),
            )
        )

        ports = sep.join(
            f"{_indent}.{port.name:<{ max_len_name +2}}"
            f"({port.get_wire_name():<{ max_len_wire +2}}) /* {port} */"
            for port in self.iter_ports(io=io, ignores=ignores)
        )
        return ports

    def render_instance(
        self,
        instnace_name: str = None,
        indent: int = INDENT,
        params: dict = None,
        io: IO = None,
    ) -> str:
        _indent = indent * " "
        params = params or self.params_redefined

        if params and isinstance(params, dict):
            params = ",\n".join(f"{_indent}.{k}({v})" for k, v in params.items())
            params = f"#(\n{params}\n) "

        if instnace_name is None:
            instnace_name = f"{self.module_name}_{self.instance_counts}"
            self.instance_counts += 1

        ports = self.render_port_connections(
            io=io,
            indent=indent,
        )

        return f"{self.module_name} {params}{instnace_name} (\n" f"{ports}\n" f");\n"

    @classmethod
    def from_file(
        cls,
        filename: str,
        modulename: str = None,
        params: PARAMS = None,
        defines: PARAMS = None,
        user_defined_functions: USER_DEFINED_FUNCTIONS = None,
        io: IO = None,
    ) -> "VerilogModule":
        filepath = Path(os.path.expandvars(filename)).expanduser().resolve()
        if not filepath.exists():
            raise FileNotFoundError(filepath)
        if modulename is None:
            modulename = filepath.stem

        parser = get_verible_verilog_parser()
        file_data = parser.parse_file(
            str(filepath), {"skip_null": True, "gen_rawtokens": False}
        )
        pp = VerilogPreprocessor(str(filepath), file_data, defines=defines)
        for module in pp.walk_root():
            _, name = cls.get_module_header_and_name(module)
            if modulename == name:
                module = cls(
                    str(filepath),
                    module,
                    defines=pp.defines,
                    params=params,
                    user_defined_functions=user_defined_functions,
                    io=io,
                )
                return module
        else:
            raise ValueError(
                f"No such module name: {modulename} in the file 'f{filepath}'"
            )

    @classmethod
    def iter_from_string(
        cls,
        string: str,
        modulename: str = None,
        params: PARAMS = None,
        defines: PARAMS = None,
        user_defined_functions: USER_DEFINED_FUNCTIONS = None,
        io: IO = None,
    ):
        parser = get_verible_verilog_parser()
        file_data = parser.parse_string(
            string, {"skip_null": True, "gen_rawtokens": False}
        )
        pp = VerilogPreprocessor("from_string", file_data, defines)
        for module in pp.walk_root():
            if modulename:
                _, name = cls.get_module_header_and_name(module)
            else:
                name = modulename
            if modulename == name:
                yield cls(
                    "from_string",
                    module,
                    defines=pp.defines,
                    params=params,
                    user_defined_functions=user_defined_functions,
                    io=io,
                )
                if modulename:
                    break
        else:
            if modulename:
                raise ValueError(f"No such module name: {modulename}")

    @classmethod
    def from_string(
        cls,
        string: str,
        modulename: str = None,
        params: PARAMS = None,
        defines: PARAMS = None,
        user_defined_functions: USER_DEFINED_FUNCTIONS = None,
        io: IO = None,
    ) -> List["VerilogModule"]:
        return list(
            cls.iter_from_string(
                string,
                modulename,
                params=params,
                defines=defines,
                user_defined_functions=user_defined_functions,
                io=io,
            )
        )