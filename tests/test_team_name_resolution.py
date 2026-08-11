"""Checks the three-way split for team names that aren't byte-for-byte
canonical:

  spec ambiguity -- the name matches one of context_pack.md's own team
      section headers verbatim (TEAM_SPEC_AMBIGUITY, e.g. 'BoomBoys /
      BetBoom Team'). prompt.md says to use "the exact team names listed
      in the briefing", but the briefing itself presents these particular
      teams under two textual forms, so this is a gap in our spec, not a
      model error. The name resolves, but parse_level must stay 1 and the
      occurrence is recorded separately for README disclosure.
  synonym -- the name matches the general legacy/shorthand table
      (TEAM_SYNONYMS, e.g. plain 'Yandex' or standalone 'BetBoom Team').
      Nothing in the prompt asked for this normalization, so it's a real
      repair: parse_level must go to 2.
  unresolved -- the name matches nothing at all. This used to sit silently
      in unknown_teams without affecting parse_level, hiding a team that's
      effectively lost to every canonical-name lookup downstream (ensemble,
      spread). It must now also push parse_level to 2.

Run with: py tests/test_team_name_resolution.py
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
    ok = True

    print("_resolve_team_name(): the three real forms confirmed in runs/")
    ok &= check("'BoomBoys / BetBoom Team' -> spec_ambiguity",
                score._resolve_team_name("BoomBoys / BetBoom Team") == ("BoomBoys", "spec_ambiguity"))
    ok &= check("'Team Vision / PARIVISION' -> spec_ambiguity",
                score._resolve_team_name("Team Vision / PARIVISION") == ("Team Vision", "spec_ambiguity"))
    ok &= check("'HULIGANI (ex-L1GA TEAM)' -> spec_ambiguity",
                score._resolve_team_name("HULIGANI (ex-L1GA TEAM)") == ("HULIGANI", "spec_ambiguity"))
    ok &= check("standalone 'BetBoom Team' -> synonym (not spec_ambiguity -- not a section header)",
                score._resolve_team_name("BetBoom Team") == ("BoomBoys", "synonym"))
    ok &= check("'Team Yandex' -> exact", score._resolve_team_name("Team Yandex") == ("Team Yandex", "exact"))
    ok &= check("'team yandex' (case only) -> case",
                score._resolve_team_name("team yandex") == ("Team Yandex", "case"))
    ok &= check("'Natus Vincere' (hallucinated, matches nothing) -> unresolved",
                score._resolve_team_name("Natus Vincere") == ("Natus Vincere", "unresolved"))

    print("real GPT data: spec-ambiguity names resolve without downgrading parse_level")
    gpt_runs = [r for r in score.load_runs(Path("runs")) if r.model == "gpt_sol5_6"]
    if not gpt_runs:
        print("  [SKIP] no runs/gpt_sol5_6_run_*.json present right now")
    else:
        for run in gpt_runs:
            ok &= check(f"{run.path.name}: parse_level == 1 despite the 3 section-header names",
                        run.parse_level == 1)
            ok &= check(f"{run.path.name}: all 16 teams resolved (no missing/unknown)",
                        not run.missing_teams and not run.unknown_teams)
            ok &= check(f"{run.path.name}: exactly 3 spec ambiguities logged",
                        len(run.spec_ambiguities) == 3)

    print("unresolved-name fixture: a genuinely unresolvable name pushes parse_level to 2")
    runs = score.load_runs(FIXTURES / "unresolved_name")
    ok &= check("fixture loaded", len(runs) == 1)
    r = runs[0]
    ok &= check("ok=True (still usable -- 15 of 16 teams are fine)", r.ok)
    ok &= check("parse_level == 2 (this is the bug fix -- it used to silently stay 1)",
                r.parse_level == 2)
    ok &= check("'Natus Vincere' surfaces in unknown_teams", "Natus Vincere" in r.unknown_teams)
    ok &= check("'GamerLegion' surfaces in missing_teams", "GamerLegion" in r.missing_teams)
    ok &= check("no repairs logged (nothing was mechanically fixed -- it just wasn't resolved)",
                r.repairs == [])
    ok &= check("no spec_ambiguities logged (this isn't a section-header case)",
                r.spec_ambiguities == [])

    report = score.coherence_report(r)
    ok &= check("coherence_report marks it a format_violation", report["format_violation"] is True)

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
