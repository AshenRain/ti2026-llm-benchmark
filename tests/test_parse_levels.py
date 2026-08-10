"""Checks the three-tier parser: level 1 (valid, exact schema), level 2
(mechanically recovered -- markdown fences/prose, key aliases, team-name
synonyms -- usable but flagged as a format violation), level 3
(unrecoverable, excluded from scoring).

tests/fixtures/parse_levels/parse-fixture_{1,2,3}.json simulate three runs
of one model:

  run 1 - level 1: exactly the schema in prompt.md, no repairs needed
  run 2 - level 2: same underlying numbers as run 1, but wrapped in
          markdown fences + prose, using an alternate-but-consistent key
          schema throughout (predictions/name/buckets/pChampion, bucket
          keys as bare "4-0" win-loss notation), and two teams under
          legacy org tags (BetBoom Team, PARIVISION)
  run 3 - level 3: truncated mid-response (as if it hit a token limit),
          no closing brace anywhere, nothing to extract

Run with: py tests/test_parse_levels.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import score  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "parse_levels"


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main() -> int:
    runs = {r.run_id: r for r in score.load_runs(FIXTURES)}
    assert set(runs) == {"1", "2", "3"}, f"expected 3 fixture runs, found {sorted(runs)}"
    r1, r2, r3 = runs["1"], runs["2"], runs["3"]

    ok = True

    print("run 1: level 1 (valid JSON, exact schema)")
    ok &= check("ok=True", r1.ok)
    ok &= check("parse_level == 1", r1.parse_level == 1)
    ok &= check("no repairs logged", r1.repairs == [])
    ok &= check("all 16 teams present", set(r1.teams) == set(score.TEAMS))

    print("run 2: level 2 (mechanically recovered, flagged as a format violation)")
    ok &= check("ok=True (usable despite needing repair)", r2.ok)
    ok &= check("parse_level == 2", r2.parse_level == 2)
    ok &= check("repairs were logged", len(r2.repairs) > 0)
    ok &= check("all 16 teams present (aliases + synonyms all resolved)",
                set(r2.teams) == set(score.TEAMS))
    ok &= check("JSON-extraction repair logged (markdown fences + prose)",
                any("extracted JSON object" in r for r in r2.repairs))
    ok &= check("top-level key alias logged ('predictions' -> 'teams')",
                any("'predictions'" in r and "'teams'" in r for r in r2.repairs))
    ok &= check("per-entry key aliases logged (name/buckets/pChampion)",
                any("'name'" in r for r in r2.repairs)
                and any("'buckets'" in r for r in r2.repairs)
                and any("'pChampion'" in r for r in r2.repairs))
    ok &= check("bucket key alias logged ('4-0' -> 'w4_l0')",
                any("'4-0'" in r and "'w4_l0'" in r for r in r2.repairs))
    ok &= check("both legacy team names resolved",
                any("BetBoom Team" in r and "BoomBoys" in r for r in r2.repairs)
                and any("PARIVISION" in r and "Team Vision" in r for r in r2.repairs))

    print("run 2 vs run 1: repairs preserve the underlying numbers")
    swiss_diffs = [
        abs(r1.teams[t]["swiss"][b] - r2.teams[t]["swiss"][b])
        for t in score.TEAMS for b in score.BUCKETS
    ]
    ok &= check("all swiss values match run 1 exactly", max(swiss_diffs) < 1e-9)
    champ_diffs = [abs(r1.teams[t]["p_champion"] - r2.teams[t]["p_champion"]) for t in score.TEAMS]
    ok &= check("p_champion values match run 1 exactly", max(champ_diffs) < 1e-9)

    print("run 3: level 3 (unrecoverable)")
    ok &= check("ok=False", not r3.ok)
    ok &= check("parse_level == 3", r3.parse_level == 3)
    ok &= check("teams is empty (nothing usable)", r3.teams == {})
    ok &= check("parse_error is set", r3.parse_error is not None)

    print("parse_summary()")
    summary = score.parse_summary(list(runs.values()))["parse-fixture"]
    ok &= check("n_total == 3", summary["n_total"] == 3)
    ok &= check("n_level1 == 1", summary["n_level1"] == 1)
    ok &= check("n_level2 == 1", summary["n_level2"] == 1)
    ok &= check("n_level3 == 1", summary["n_level3"] == 1)
    ok &= check("n_usable == 2 (level 1 + level 2)", summary["n_usable"] == 2)

    print("scoring: level 2 is included in scoring and matches level 1's result")
    results = {t: "w4_l2" for t in score.TEAMS}  # arbitrary fixed outcome, same for both checks
    brier1 = score.multiclass_brier(r1.teams, results)
    brier2 = score.multiclass_brier(r2.teams, results)
    ok &= check("level-2 run scores (not skipped, not None)", brier2 is not None)
    ok &= check("level-2 run's score matches level-1's (repairs preserved the numbers)",
                abs(brier1 - brier2) < 1e-9)

    print("scoring: strict (level-1-only) vs all (level-1+level-2) aggregation")
    level1_runs = [r for r in runs.values() if r.parse_level == 1]
    usable_runs = [r for r in runs.values() if r.ok]
    ok &= check("strict pool excludes the level-2 run", len(level1_runs) == 1 and r2 not in level1_runs)
    ok &= check("lenient pool includes both level-1 and level-2 runs",
                len(usable_runs) == 2 and r1 in usable_runs and r2 in usable_runs)
    ok &= check("unrecoverable run excluded from both pools", r3 not in level1_runs and r3 not in usable_runs)

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
