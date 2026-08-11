"""Checks plot_column_sums() runs and produces a real, deterministic PNG.
Never writes to the project's charts/ directory -- always a temp path, so
this test can't clobber the checked-in chart.

Run with: py tests/test_charts.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import score  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main() -> int:
    ok = True

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        print("plot_column_sums(): produces a real PNG from the parse-level fixtures")
        runs = score.load_runs(FIXTURES / "parse_levels")
        out1 = tmp_path / "one.png"
        score.plot_column_sums(runs, out1)
        ok &= check("output file exists", out1.exists())
        ok &= check("output is a valid PNG (magic bytes)",
                     out1.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n")
        ok &= check("output is a non-trivial size (not a blank/broken render)",
                     out1.stat().st_size > 20_000)

        print("plot_column_sums(): deterministic given the same input (fixed jitter seed)")
        out2 = tmp_path / "two.png"
        score.plot_column_sums(runs, out2)
        ok &= check("re-running against the same runs produces byte-identical output",
                     out1.read_bytes() == out2.read_bytes())

        print("plot_column_sums(): creates its parent directory if missing")
        nested = tmp_path / "nested" / "dir" / "chart.png"
        score.plot_column_sums(runs, nested)
        ok &= check("nested output directory was created", nested.exists())

        print("plot_column_sums(): unrecoverable (level-3) runs are excluded, not crashed on")
        mixed_dir = FIXTURES / "unresolved_name"
        mixed_runs = score.load_runs(mixed_dir)
        out3 = tmp_path / "mixed.png"
        try:
            score.plot_column_sums(mixed_runs, out3)
            ok &= check("renders without error even with a format-violation run", out3.exists())
        except Exception as exc:  # noqa: BLE001
            ok = check(f"renders without error even with a format-violation run (raised {exc!r})", False)

        print("plot_column_sums(): real runs/ data, if present, renders too")
        real_runs_dir = Path("runs")
        if real_runs_dir.exists() and any(real_runs_dir.glob("*.json")):
            real_runs = score.load_runs(real_runs_dir)
            out4 = tmp_path / "real.png"
            score.plot_column_sums(real_runs, out4)
            ok &= check("renders against the full real dataset", out4.exists() and out4.stat().st_size > 20_000)
        else:
            print("  [SKIP] no runs/*.json present right now")

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
