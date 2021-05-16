import math
from typing import Union


def clog2(value: Union[int, str]):
    return math.ceil(math.log2(int(value)))


def flog2(value: Union[int, str]):
    return math.floor(math.log2(int(value)))


def vbin(value: Union[int, str], size: int = None):
    """
    return verilog binary format
    """
    binary = bin(int(value))[2:]
    if not size:
        size = len(binary)
    return f"{size}'b{binary[-size:]}"


def vhex(value: Union[int, str], size: int = None):
    """
    return verilog hex format
    """
    binary = bin(int(value))[2:]
    if not size:
        size = len(binary)
    return f"{size}'h{int(binary[-size:], 2):0X}"


def voct(value: Union[int, str], size: int = None):
    """
    return verilog octal format
    """
    binary = bin(int(value))[2:]
    if not size:
        size = len(binary)
    return f"{size}'o{int(binary[-size:], 2):0o}"


def vdec(value: Union[int, str], size: int = None):
    """
    return verilog decimal format
    """
    binary = bin(int(value))[2:]
    if not size:
        size = len(binary)
    return f"{size}'d{int(binary[-size:], 2)}"
