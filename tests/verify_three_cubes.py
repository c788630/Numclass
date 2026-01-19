"""
verify_three_cubes.py — validate stored solutions for x^3 + y^3 + z^3 = n

Default behavior:
  - Loads the new-format TOML from the Numclass workspace:
        <Documents>/Numclass/data/sum_of_three_cubes.toml
    falling back to the packaged copy if needed.

Options:
  --file PATH        : validate a specific TOML file (new [solutions] format)
  --canonicalize     : print canonicalized triples for valid entries
  --emit-json PATH   : write a JSON report to PATH

Run:
  python -m tests.verify_three_cubes
  python -m tests.verify_three_cubes --canonicalize
  python -m tests.verify_three_cubes --file path/to/sum_of_three_cubes.toml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Numclass loaders (workspace + packaged fallback)
from numclass.utility import load_sum_of_three_cubes_toml
from numclass.workspace import workspace_dir

try:
    import tomllib as _toml  # py311+
except Exception:  # pragma: no cover
    import tomli as _toml  # type: ignore


def _parse_new_format_toml(text: str) -> dict[int, tuple[int, int, int]]:
    """
    Strict parser for the new schema:

        [solutions]
        42 = [-80538738812075974, 80435758145817515, 12602123297335631]
        33 = [8866128975287528, -8778405442862239, -2736111468807040]
        ...

    Returns { n: (a,b,c) }.
    """
    try:
        data = _toml.loads(text)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    sol = data.get("solutions")
    if not isinstance(sol, dict):
        return {}

    out: dict[int, tuple[int, int, int]] = {}
    for k, v in sol.items():
        try:
            n = int(k)
        except Exception:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 3:
            try:
                a, b, c = int(v[0]), int(v[1]), int(v[2])
            except Exception:
                continue
            out[n] = (a, b, c)
    return out


def _load_from_file(path: Path) -> dict[int, tuple[int, int, int]]:
    text = path.read_text(encoding="utf-8")
    return _parse_new_format_toml(text)


def sum_of_cubes(triple: tuple[int, int, int]) -> int:
    x, y, z = triple
    return x**3 + y**3 + z**3


def canonicalize(triple: tuple[int, int, int]) -> tuple[int, int, int]:
    # sort by absolute value descending; for ties, natural order
    x, y, z = triple
    arr = sorted([x, y, z], key=lambda t: (abs(t), t), reverse=True)
    return tuple(arr)  # type: ignore[return-value]


def verify_solutions(solutions: dict[int, tuple[int, int, int]]):
    report = {
        "total": len(solutions),
        "valid": 0,
        "invalid": 0,
        "entries": [],  # dicts: {n, triple, sum, ok, diff, canonical}
        "source": None,
    }
    for n, triple in sorted(solutions.items()):
        s = sum_of_cubes(triple)
        ok = (s == n)
        report["entries"].append(
            {
                "n": n,
                "triple": triple,
                "sum": s,
                "ok": ok,
                "diff": int(s - n),
                "canonical": canonicalize(triple),
            }
        )
        report["valid"] += int(ok)
        report["invalid"] += int(not ok)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file",
        dest="file",
        help="TOML file to validate (new [solutions] format). If omitted, uses the Numclass workspace default.",
    )
    ap.add_argument("--canonicalize", action="store_true", help="Print canonicalized triples for valid entries.")
    ap.add_argument("--emit-json", dest="emit_json", help="Write JSON report to this path.")
    args = ap.parse_args()

    # Load mapping {n: (a,b,c)}
    source_desc: str
    if args.file:
        p = Path(args.file)
        if not p.is_file():
            print(f"File not found: {p}")
            return 2
        solutions = _load_from_file(p)
        source_desc = str(p)
    else:
        # Workspace default + packaged fallback
        solutions = load_sum_of_three_cubes_toml("sum_of_three_cubes.toml")
        source_desc = str(workspace_dir() / "data" / "sum_of_three_cubes.toml")

    report = verify_solutions(solutions)
    report["source"] = source_desc

    print(f"Verified sum-of-3-cubes solutions from: {source_desc}")
    print(f"Total entries: {report['total']}  |  Valid: {report['valid']}  |  Invalid: {report['invalid']}")

    if report["invalid"]:
        print("\nInvalid entries:")
        for e in report["entries"]:
            if not e["ok"]:
                n = e["n"]
                x, y, z = e["triple"]
                s = e["sum"]
                diff = e["diff"]
                print(f"  n={n}: ({x}, {y}, {z})  => sum={s}  (diff={diff})")

    if args.canonicalize:
        print("\nCanonicalized valid entries:")
        for e in report["entries"]:
            if e["ok"]:
                n = e["n"]
                a, b, c = e["canonical"]
                print(f"  {n}: ({a}, {b}, {c})")

    if args.emit_json:
        outp = Path(args.emit_json)
        outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {outp}")

    # Non-zero exit if invalid entries found
    return 1 if report["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
