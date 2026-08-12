"""P1: the global atom index — `atom_id -> (module, layer, matrix_type, c)`.

Everything downstream keys on `atom_id`, defined by one canonical order fixed here:
layer-major, computation order within a layer (q, k, v, o, mlp.in, mlp.out). The order
is a function of `module_to_c` only, so any loaded model reproduces it exactly.
"""

import re

import pandas as pd

KIND_ORDER = ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj")

MATRIX_TYPE = {
    "attn.q_proj": "attn.q",
    "attn.k_proj": "attn.k",
    "attn.v_proj": "attn.v",
    "attn.o_proj": "attn.o",
    "mlp.c_fc": "mlp.in",
    "mlp.down_proj": "mlp.out",
}

_MODULE_RE = re.compile(r"^h\.(\d+)\.(attn|mlp)\.(\w+)$")


def canonical_modules(module_to_c: dict[str, int]) -> list[str]:
    """Module paths in canonical atom order. Asserts every path matches h.<L>.<sub>.<kind>."""

    def key(path: str) -> tuple[int, int]:
        m = _MODULE_RE.match(path)
        assert m, f"Unexpected module path: {path}"
        layer = int(m.group(1))
        kind = f"{m.group(2)}.{m.group(3)}"
        assert kind in KIND_ORDER, f"Unknown kind {kind} in {path}"
        return layer, KIND_ORDER.index(kind)

    return sorted(module_to_c, key=key)


def build_atom_index(module_to_c: dict[str, int]) -> pd.DataFrame:
    """One row per atom: atom_id, module, layer, matrix_type, c."""
    rows = []
    atom_id = 0
    for module in canonical_modules(module_to_c):
        m = _MODULE_RE.match(module)
        assert m is not None
        layer = int(m.group(1))
        matrix_type = MATRIX_TYPE[f"{m.group(2)}.{m.group(3)}"]
        for c in range(module_to_c[module]):
            rows.append((atom_id, module, layer, matrix_type, c))
            atom_id += 1
    return pd.DataFrame(rows, columns=["atom_id", "module", "layer", "matrix_type", "c"])


def module_slices(module_to_c: dict[str, int]) -> dict[str, slice]:
    """Column slice of each module inside the canonically ordered [.., A] atom axis."""
    slices: dict[str, slice] = {}
    start = 0
    for module in canonical_modules(module_to_c):
        c = module_to_c[module]
        slices[module] = slice(start, start + c)
        start += c
    return slices
