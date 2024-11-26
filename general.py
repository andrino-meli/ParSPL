import sys
import os
import json
import numpy as np

# from bfloat16 import bfloat16
from typing import Any
from matplotlib import rcParams

# default font
rcParams["font.size"] = 13  # Example: 12 for font size
rcParams["font.family"] = "Times New Roman"
# Configure Matplotlib to embed fonts in the SVG
rcParams["svg.fonttype"] = "none"


DEBUG = False
THRESHINF = 1e17  # above this threshold any value is set to infinity
INF = 1e17  # constant that is used for Infinity

NUM_CORES = 8
WORKERS = NUM_CORES
HARTS = NUM_CORES
GRAY_COLOR = (0.3, 0.3, 0.3)

color_palette = [
    (0x1E, 0x4A, 0x28),
    (0x51, 0xA1, 0x6A),
    (0x77, 0xAB, 0x75),
    (0x9C, 0xB5, 0x7F),
    (0xE6, 0xC9, 0x94),
    (0xCC, 0x7A, 0x3D),
    (0xC8, 0x67, 0x39),
    (0xC4, 0x53, 0x35),
]
color_palette = [(r / 255, g / 255, b / 255) for (r, g, b) in color_palette]
color_palette.reverse()


def list_flatten(data):
    tmp = []
    assert isinstance(data, list)
    for l in data:
        if isinstance(l, list):
            tmp.extend(list_flatten(l))
        else:
            tmp.append(l)
    return tmp


class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __setitem__(self, key: Any, item: Any) -> None:
        if isinstance(item, dict):
            item = DotDict(item)
        super().__setitem__(key, item)

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(key)

    def __setattr__(self, key: Any, item: Any) -> None:
        return self.__setitem__(key, item)

    def __getattr__(self, key: Any) -> Any:
        return self.__getitem__(key)

    def __missing__(self, key: Any) -> str:
        raise KeyError(f"DotDict object has no item '{key}'")

    def __getstate__(self) -> dict:
        return self

    def __setstate__(self, state: dict) -> None:
        self.update(state)
        self.__dict__ = self

    def __delattr__(self, item: Any) -> None:
        self.__delitem__(item)

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)


def primifyDotDict(dd: DotDict):
    """Convert numpy arrays to primitive types."""
    for k, v in dd.items():
        if isinstance(v, np.ndarray):
            wprint(f"Convert {k} form numpy array to list")
            dd[k] = v.tolist()


def dumpDotDict(dd: DotDict, filename: str):
    primifyDotDict(dd)
    try:
        json.dump(dd, open(filename, "w"), indent=0)
    except Exception as e:
        if os.path.exists(filename):
            eprint(f'Hit exception "{e}". Deleting {filename}.')
            os.remove(filename)


def loadDotDict(filename: str):
    dd = json.load(open(filename))
    return DotDict(**dd)


def type_to_str(t):
    if t is np.float64:
        dtype = "double"
        # print(f'Warning used float64 in data.')
    elif t is np.float32:
        dtype = "float"
    elif t is np.float16:
        dtype = "__fp16"
    elif t is np.int64:
        dtype = "int64_t"
    elif t is np.uint64:
        dtype = "uint64_t"
    elif t is np.int32:
        dtype = "int32_t"
    elif t is np.uint32:
        dtype = "uint32_t"
    elif t is np.int16:
        dtype = "int16_t"
    elif t is np.uint16:
        dtype = "uint16_t"
    elif t is np.int8:
        dtype = "int8_t"
    elif t is np.uint8:
        dtype = "uint8_t"
    elif t is bfloat16:
        dtype = "__fp16alt"
    else:
        raise ValueError(f"Unexpected Type {t}.")
    return dtype


def svdinvert(A):
    u, s, v = np.linalg.svd(A)
    Ainv = np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose()))
    return Ainv


def max_rel_err(x, gold, tol=1e-3, log=False):
    x = x.flatten()
    gold = gold.flatten()
    err = np.abs(x - gold)
    relerr = err / gold
    idx = np.argmax(relerr)
    m = relerr[idx]
    if m > tol:
        eprint(
            escape.RED,
            f"FUNCTIONAL ERROR: Max rel error = {m:.4e} at index {idx}",
            escape.END,
        )
        eprint(f"Calculated {x[idx]} but want {gold[idx]}.")
    elif log:
        print(f"Max rel error = {m:.4e}")
        # eprint(f'x =    {x}\ngold = {gold}')
    return relerr


class escape:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def eprint(*args, **kwargs):
    print(escape.RED, *args, escape.END, file=sys.stderr, **kwargs)


def wprint(*args, **kwargs):
    print(escape.YELLOW, *args, escape.END, file=sys.stderr, **kwargs)


def kib(x):
    return f"{x/2**10:.1f} KiB"


def percent(x):
    return f"{100*x:4.1f}%"


def bprint(*args, **kwargs):
    # print(escape.BOLD,end=None,sep=None)
    print(escape.BOLD, *args, escape.END, **kwargs)


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, file=sys.stderr, **kwargs)


def list2array(data, name, base=None):
    """Convert a list to a numpy unsigned int array.
    If base is given choose type accordingly."""
    assert isinstance(data, list)
    assert len(data) != 0
    assert base is None or base == 8 or base == 16 or base == 32
    maximum = int(max(data))
    minimum = int(min(data))
    if minimum < 0:
        wprint(f"Warning: {name} contains negative values: overflowing")
    maximum = max(maximum, -minimum)
    typ = None
    if base is None:
        if maximum < 2**8:
            typ = np.uint8
        elif maximum < 2**16:
            typ = np.uint16
        elif maximum < 2**32:
            typ = np.uint32
    else:
        if base == 8:
            typ = np.uint8
            if maximum >= 2**8:
                raise ValueError(
                    f"Requested to store {name} in {typ} but maximum {maximum} will overflow."
                )
        elif base == 16:
            typ = np.uint16
            if maximum >= 2**16:
                raise ValueError(
                    f"Requested to store {name} in {typ} but maximum {maximum} will overflow."
                )
        elif base == 32:
            typ = np.uint32
            if maximum >= 2**32:
                raise ValueError(
                    f"Requested to store {name} in {typ} but maximum {maximum} will overflow."
                )
    if typ is None:
        raise NotImplementedError(
            f"Data Type required to store {maximum} is not implemented."
        )
    dprint(f"Storing {name} with {type_to_str(typ)}")
    return np.array(data, dtype=typ)


# Dump numpy nd array to file
def ndarrayToC(f, name, arr, section=".tcdm", const=False):
    const = "const " if const else ""
    if arr.dtype.kind == "f":
        al = 8  # allignment of 64-bit double
    else:
        al = 4  # allignment of 32-bit
    if section is None or len(section) == 0:
        attr = f"__attribute__((aligned({al})))"
    else:
        attr = f'__attribute__((aligned({al}),section("{section}")))'
    sh = "[" + str(arr.shape[0])
    for s in arr.shape[1:]:
        sh += "*" + str(s)
    sh += "]"
    arr = arr.flatten()
    dtype = type_to_str(type(arr[0]))
    f.write(f"{const}{dtype} {name:>10} {sh} {attr} = " + "{\n")
    for i, v in enumerate(arr):
        f.write(f"{v}, ")
        if (i + 1) % 5 == 0:
            f.write("\n")
    f.write("};\n\n")
    return (dtype, sh)


def ndarrayToCH(fc, fh, name, arr, section=".tcdm", const=False):
    dtype, sh = ndarrayToC(fc, name, arr, section=section, const=const)
    const = "const " if const else ""
    fh.write(f"extern {const}{dtype} {name:>10} {sh};\n")
