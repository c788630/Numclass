# run_scan_sum_of_three_cubes.py
from pathlib import Path
from datetime import datetime
import sys

from user import settings
from classifiers.arithmetic_divisor import is_sum_of_3_cubes

# Make sure the brute-force range is exactly [-100, 100]
setattr(settings, "MAX_ABS_FOR_SUM_OF_3_CUBES", 100)

# Optional: allow a single result per n (speeds things up)
setattr(settings, "MAX_SOLUTIONS_SUM_OF", {"3_CUBES": 1})

# Optional: whether to allow zeros in (a,b,c)
setattr(settings, "ALLOW_ZERO_IN_DECOMP", True)


def main():
    need_update = []  # n for which is_sum_of_3_cubes(...) returned False
    checked = 0
    total_n = 1000

    for n in range(1, total_n + 1):
        # progress line (fixed width so shorter strings get overwritten)
        sys.stdout.write(f"\rScanning n={n:4d}/{total_n} ...")
        sys.stdout.flush()

        if n % 9 in (4, 5):
            continue  # impossible by the mod-9 obstruction

        ok, _details = is_sum_of_3_cubes(n, max_results=1)  # uses limit = 100 via settings
        checked += 1
        if not ok:
            need_update.append(n)

    # newline to end the progress line cleanly
    sys.stdout.write("\n")
    sys.stdout.flush()

    out_path = Path("needs_update_sum_of_3_cubes_1_1000.txt")
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# n in [1,1000] (excluding 4,5 mod 9) for which is_sum_of_3_cubes returned False\n")
        f.write(f"# MAX_ABS_FOR_SUM_OF_3_CUBES = {getattr(settings, 'MAX_ABS_FOR_SUM_OF_3_CUBES', None)}\n")
        f.write(f"# generated {datetime.utcnow().isoformat()}Z\n\n")
        for n in need_update:
            f.write(f"{n}\n")

    print(f"Checked {checked} values; wrote {len(need_update)} numbers to {out_path.resolve()}")


if __name__ == "__main__":
    main()
