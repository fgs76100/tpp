#!/usr/bin/env python3

import os

from pathlib import Path
from fnmatch import fnmatch
from typing import Union, Callable, Generator
from .core import (
    VerilogModuleParser,
    VerilogModulePort,
    VerilogPreprocessor,
    verible_verilog_parser,
)
from .verible_verilog_syntax import BranchNode

INDENT = 2
UNCONNECTED = 0
INTERFACE = 1


def rename(string: str, repl: Union[dict, Callable]):
    if callable(repl):
        new = repl(string)
    elif repl and isinstance(repl, dict):
        new = repl.get(string, string)
    else:
        new = string
    return new


class VerilogModule(VerilogModuleParser):
    def __init__(
        self,
        file_path: str,
        module_tree: BranchNode,
        defines: dict = None,
        params: dict = None,
        io_suffix: str = "",
        io_prefix: str = "",
        io_mode: int = INTERFACE,
        io_conn: dict = None,
    ):
        super().__init__(file_path, module_tree, defines, params)
        self.instance_counts = 0
        self.io_conn = io_conn or {}
        self.io_suffix = io_suffix
        self.io_prefix = io_prefix
        self.set_io_mode(io_mode)
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
        match: Union[str, Callable] = None,
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
            if ignores and port.name in ignores:
                continue

            if callable(match) and not match(port.name):
                continue

            if match and isinstance(match, str):
                if not fnmatch(port.name, match):
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

            yield port

    def iter_declarations(self, data_type: str, ignores: list = None):
        max_len_dimension = max(
            map(
                lambda x: len(str(x.data_type.to_declaration(data_type))),
                self.iter_ports(),
            )
        )
        for port in self.iter_ports(ignores=ignores):
            if (
                self.io_mode == UNCONNECTED
                and port.direction == "input"
                and port.is_unconnected
            ):
                yield f"{port.to_declaration(data_type, max_len_dimension)} = {port.data_type.size}'h0;"
            else:
                yield f"{port.to_declaration(data_type, max_len_dimension)};"

    def io_to_wires(self, ignores: list = None):
        return "\n".join(self.iter_declarations("wire", ignores))

    def io_to_logics(self, ignores: list = None):
        return "\n".join(self.iter_declarations("logic", ignores))

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