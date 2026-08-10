"""Checks that bookmaker/market sources are kept separate rather than
merged, and that entrants lacking bucket-level data (any market baseline,
which only prices the champion) get an explicit "N/A" (None), never a
silent 0 or a silently dropped row.

tests/fixtures/odds_two_sources.csv has two teams priced by two sources:
  srcA: Team A @ 2.00, Team B @ 4.00  (overround: 1/2 + 1/4 = 0.75)
  srcB: Team A @ 3.00, Team B @ 3.00  (overround: 1/3 + 1/3 = 0.667)

Run with: py tests/test_baselines.py
"""

import math
import sys
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

    print("bookmaker_baselines: sources kept separate")
    baselines = score.bookmaker_baselines(FIXTURES / "odds_two_sources.csv")
    ok &= check("returns both sources, not merged into one", set(baselines) == {"srcA", "srcB"})

    srcA, srcB = baselines["srcA"], baselines["srcB"]
    ok &= check("srcA normalized within its own group (sums to 1.0)",
                abs(sum(v["p_champion"] for v in srcA.values()) - 1.0) < 1e-9)
    ok &= check("srcB normalized within its own group (sums to 1.0)",
                abs(sum(v["p_champion"] for v in srcB.values()) - 1.0) < 1e-9)
    ok &= check("srcA Team A = 0.6667 (1/2 of 0.75 overround, normalized)",
                abs(srcA["Team A"]["p_champion"] - 2 / 3) < 1e-6)
    ok &= check("srcB is 50/50 (equal odds), unaffected by srcA's numbers",
                abs(srcB["Team A"]["p_champion"] - 0.5) < 1e-9)
    ok &= check("every team's swiss is None (no bucket data from odds)",
                all(v["swiss"] is None for v in srcA.values())
                and all(v["swiss"] is None for v in srcB.values()))

    print("scoring functions: N/A for bucket-level metrics on a bookmaker entrant")
    results = {"Team A": "w4_l0", "Team B": "w1_l4"}
    ok &= check("multiclass_brier returns None (not 0, not a crash)",
                score.multiclass_brier(srcA, results) is None)
    ok &= check("multiclass_logloss returns None",
                score.multiclass_logloss(srcA, results) is None)
    ok &= check("binary_advance_brier returns None",
                score.binary_advance_brier(srcA, results) is None)

    champ_score = score.champion_brier(srcA, "Team A")
    ok &= check("champion_brier still computes a real number (champion IS priced)",
                champ_score is not None and not math.isnan(champ_score))
    expected = ((2 / 3 - 1.0) ** 2 + (1 / 3 - 0.0) ** 2) / 2
    ok &= check("champion_brier value matches hand-computed Brier score",
                abs(champ_score - expected) < 1e-6)

    print("scoring functions: a fully-specified entrant (uniform) is unaffected")
    uniform = score.uniform_baseline()
    full_results = {t: "w4_l2" for t in score.TEAMS}
    ok &= check("multiclass_brier is a real number for uniform baseline",
                score.multiclass_brier(uniform, full_results) is not None)

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
