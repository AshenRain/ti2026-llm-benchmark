"""Checks the scale-diagnostics table: every probability column's sum is
shown, raw columns (which sum above 1.0 by design -- that's the platform's
margin) are never flagged, and a column claimed to be normalized that
doesn't actually sum to its expected total raises an explicit, loud error
rather than passing silently.

This guards against exactly the bug that prompted it: comparing a raw
implied probability (e.g. Team Yandex at 1/6.00 = 0.167 from one
bookmaker) against a normalized one (e.g. 0.196 from another, after
dividing by that source's own total) as if they were on the same scale.

Run with: py tests/test_scale_diagnostics.py
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import score  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
ODDS_PATH = REPO_ROOT / "odds.csv"


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main() -> int:
    ok = True

    print("_scale_check: raw columns are never flagged, even with overround")
    raw_result = score._scale_check("raw test", [0.5, 0.4, 0.3], is_normalized=False)
    ok &= check("raw column sum is just reported (1.2)", abs(raw_result["sum"] - 1.2) < 1e-9)
    ok &= check("raw column is never an error, regardless of its sum", raw_result["error"] is False)

    print("_scale_check: a correctly normalized column passes")
    good = score._scale_check("good normalized", [0.5, 0.3, 0.2], is_normalized=True, expected=1.0)
    ok &= check("no error when sum matches expected within tolerance", good["error"] is False)

    print("_scale_check: a column CLAIMED normalized but off catches the exact bug reported")
    # Team Yandex-style mismatch: values on a raw scale (0.167, 0.230) mislabeled normalized.
    bad = score._scale_check("mislabeled raw-as-normalized", [0.167, 0.230], is_normalized=True, expected=1.0)
    ok &= check("flagged as an error, not silently accepted", bad["error"] is True)

    print("_scale_check: right at the 1e-6 tolerance boundary")
    boundary_ok = score._scale_check("at tolerance", [0.5 + 5e-7, 0.5], is_normalized=True, expected=1.0)
    boundary_bad = score._scale_check("past tolerance", [0.5 + 5e-6, 0.5], is_normalized=True, expected=1.0)
    ok &= check("5e-7 deviation is within the 1e-6 tolerance", boundary_ok["error"] is False)
    ok &= check("5e-6 deviation exceeds the 1e-6 tolerance", boundary_bad["error"] is True)

    print("scale_diagnostics(): end-to-end over real bookmaker data + parse-level fixtures")
    runs = score.load_runs(FIXTURES / "parse_levels")
    checks = score.scale_diagnostics(runs, ODDS_PATH)
    by_label = {c["label"]: c for c in checks}

    raw_esportbet = next((c for label, c in by_label.items() if "esportbet" in label and "raw" in label), None)
    norm_esportbet = next((c for label, c in by_label.items() if "esportbet" in label and "normalized" in label), None)
    if raw_esportbet is None or norm_esportbet is None:
        print("  [SKIP] odds.csv has no esportbet_aggregate rows right now -- skipping real-data checks")
    else:
        ok &= check("esportbet raw column sums above 1.0 (real overround)", raw_esportbet["sum"] > 1.0)
        ok &= check("esportbet raw column is never flagged", raw_esportbet["error"] is False)
        ok &= check("esportbet normalized column sums to 1.0", abs(norm_esportbet["sum"] - 1.0) < 1e-6)
        ok &= check("esportbet normalized column has no error", norm_esportbet["error"] is False)

    ok &= check("every model-run swiss/champion column is present and normalized",
                all(c["is_normalized"] for c in checks if "model parse-fixture" in c["label"]))
    ok &= check("no unrecoverable (level-3) run contributes a column",
                not any("run 3" in c["label"] for c in checks))

    print("_print_scale_diagnostics: prints a loud banner and returns True on error")
    buf = io.StringIO()
    with redirect_stdout(buf):
        had_error = score._print_scale_diagnostics([bad, good, raw_result])
    output = buf.getvalue()
    ok &= check("returns True when a normalized column is off", had_error is True)
    ok &= check("prints an explicit SCALE ERROR banner", "SCALE ERROR" in output)

    buf_clean = io.StringIO()
    with redirect_stdout(buf_clean):
        had_error_clean = score._print_scale_diagnostics([good, raw_result])
    ok &= check("returns False when nothing is wrong", had_error_clean is False)
    ok &= check("no error banner when everything checks out", "SCALE ERROR" not in buf_clean.getvalue())

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
