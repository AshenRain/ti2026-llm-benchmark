"""Sanity check that score.py's coherence checks catch known-bad input.

tests/fixtures/fixture-model_{1,2,3}.json simulate three runs of one model,
each violating a different constraint from CLAUDE.md on purpose:

  run 1 - row-sum violation (Team Yandex) + a missing team (GamerLegion)
  run 2 - champion-column violation, an advance violation (OG), and a
          legacy team name (BetBoom Team instead of BoomBoys) that the
          synonym table resolves -- a parse-level-2 format violation, not
          a missing/unknown team
  run 3 - column-sum violation (two teams both claim a certain w4_l0), and
          the whole response wrapped in markdown fences + prose, to check
          the defensive JSON extraction (also parse level 2)

Run with: py tests/test_coherence.py
"""

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
    runs = {r.run_id: r for r in score.load_runs(FIXTURES)}
    assert set(runs) == {"1", "2", "3"}, f"expected 3 fixture runs, found {sorted(runs)}"

    ok = True

    print("run 1 (row-sum + missing team)")
    r1 = runs["1"]
    ok &= check("parses successfully", r1.ok)
    ok &= check("parse level 1 (no repairs needed -- coherence bugs aren't format bugs)",
                r1.parse_level == 1)
    ok &= check("flags GamerLegion as missing", "GamerLegion" in r1.missing_teams)
    row_dev = score.row_deviations(r1)
    ok &= check("flags Team Yandex row-sum deviation > tolerance",
                row_dev["Team Yandex"] > score.TOLERANCE)
    ok &= check("Team Yandex deviation matches expected +0.15",
                abs(row_dev["Team Yandex"] - 0.15) < 1e-9)
    col_dev = score.column_deviations(r1)
    ok &= check("column deviation nonzero (fallout from missing team)",
                max(col_dev.values()) > score.TOLERANCE)

    print("run 2 (champion-column + advance violation + legacy team name)")
    r2 = runs["2"]
    ok &= check("parses successfully", r2.ok)
    ok &= check("parse level 2 (legacy name needed synonym resolution)", r2.parse_level == 2)
    ok &= check("BoomBoys resolved, not missing", "BoomBoys" not in r2.missing_teams)
    ok &= check("BetBoom Team resolved, not left unknown", "BetBoom Team" not in r2.unknown_teams)
    ok &= check("repairs log records the synonym resolution",
                any("BetBoom Team" in r and "BoomBoys" in r for r in r2.repairs))
    champ_dev = score.champion_deviation(r2)
    ok &= check("champion column deviates far from 1.0", champ_dev > 0.5)
    violations = score.advance_violations(r2)
    violated_teams = [team for team, _, _ in violations]
    ok &= check("flags OG's advance violation (p_champion > advance sum)",
                "OG" in violated_teams)

    print("run 3 (column-sum violation + markdown-wrapped JSON)")
    r3 = runs["3"]
    ok &= check("defensively parses markdown-fenced JSON with surrounding prose", r3.ok)
    ok &= check("parse level 2 (needed JSON extraction)", r3.parse_level == 2)
    col_dev3 = score.column_deviations(r3)
    ok &= check("flags w4_l0 column-sum violation (two teams both claim it)",
                col_dev3["w4_l0"] > 1.0)
    row_dev3 = score.row_deviations(r3)
    ok &= check("rows themselves stay valid (violation isolated to the column)",
                max(row_dev3.values()) < score.TOLERANCE)
    ok &= check("champion column stays valid (violation isolated)",
                score.champion_deviation(r3) < score.TOLERANCE)

    print("cross-run spread")
    spread = score.spread_across_runs(list(runs.values()))
    ok &= check("computes a noise-floor spread despite missing/renamed teams across runs",
                spread["Team Falcons"]["p_champion"]["n"] >= 2)

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
