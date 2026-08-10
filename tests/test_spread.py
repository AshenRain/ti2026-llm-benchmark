"""Checks spread_across_runs() / noise_floor() compute the cross-run noise
floor CLAUDE.md specifies: for each team, the SAMPLE stdev (N-1) of its own
value across the model's three runs, then the mean of that over the 16
teams. NOT the spread of the 16 teams' values within a single run -- that
answers a different question (how much teams differ from each other) and
would be roughly 5-7x larger on real data (verified against
deepseek_chat_search-on: 0.070 within-run vs 0.010 the correct way). And
not population stdev (N) either: three runs are a sample used to estimate
the model's true noise, not the full population of interest -- see
CLAUDE.md's Design decisions.

tests/fixtures/spread_check/spread-fixture_run_{1,2,3}.json give every team
the identical jitter pattern [base-D, base, base+D] (D=0.03) for p_champion,
w4_l0 and w4_l1, and hold the other four Swiss buckets constant. That makes
the expected noise floor known exactly: sample stdev of [-D, 0, D] happens
to equal D exactly (a coincidence of this specific 3-point pattern, not a
general identity) for the jittered columns, 0 for the constant ones -- a
control value the formula must hit, not just "some smaller number than the
within-run spread."

Run with: py tests/test_spread.py
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import score  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "spread_check"
D = 0.03
EXPECTED_JITTERED = statistics.stdev([-D, 0, D])  # sample stdev; equals D exactly here
TOL = 1e-9


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main() -> int:
    ok = True

    runs = score.load_runs(FIXTURES)
    ok &= check("all 3 fixture runs parse at level 1 (clean control data)",
                len(runs) == 3 and all(r.parse_level == 1 for r in runs))

    print("spread_across_runs(): per-team stdev matches the hand-derived control value")
    spread = score.spread_across_runs(runs)
    for team in score.TEAMS:
        champ_stdev = spread[team]["p_champion"]["stdev"]
        if abs(champ_stdev - EXPECTED_JITTERED) > TOL:
            ok = check(f"{team}: p_champion stdev matches D (sample stdev)", False)
    ok &= check("every team's p_champion stdev matches D (sample stdev) (spot check via max deviation)",
                max(abs(spread[t]["p_champion"]["stdev"] - EXPECTED_JITTERED) for t in score.TEAMS) < TOL)
    ok &= check("every team's w4_l0 stdev matches D (sample stdev)",
                max(abs(spread[t]["swiss"]["w4_l0"]["stdev"] - EXPECTED_JITTERED) for t in score.TEAMS) < TOL)
    ok &= check("every team's w4_l1 stdev matches D (sample stdev) (mirrored jitter, same magnitude)",
                max(abs(spread[t]["swiss"]["w4_l1"]["stdev"] - EXPECTED_JITTERED) for t in score.TEAMS) < TOL)
    for bucket in ("w4_l2", "w2_l4", "w1_l4", "w0_l4"):
        ok &= check(f"every team's {bucket} stdev is exactly 0 (held constant across runs)",
                    max(spread[t]["swiss"][bucket]["stdev"] for t in score.TEAMS) == 0.0)

    print("noise_floor(): mean over 16 teams matches the same control value")
    floor = score.noise_floor(spread)
    ok &= check("p_champion noise floor == D (sample stdev)", abs(floor["p_champion"] - EXPECTED_JITTERED) < TOL)
    ok &= check("swiss[w4_l0] noise floor == D (sample stdev)", abs(floor["w4_l0"] - EXPECTED_JITTERED) < TOL)
    ok &= check("swiss[w4_l1] noise floor == D (sample stdev)", abs(floor["w4_l1"] - EXPECTED_JITTERED) < TOL)
    for bucket in ("w4_l2", "w2_l4", "w1_l4", "w0_l4"):
        ok &= check(f"swiss[{bucket}] noise floor == 0", floor[bucket] == 0.0)

    print("regression check: this is NOT within-run spread across the 16 teams")
    # Every team gets the SAME p_champion within any one run of this fixture
    # (the jitter is per-run, not per-team) -- so the wrong-axis formula
    # (stdev of the 16 teams' values within a single run) reports exactly 0
    # here, even though the model clearly has real, controlled cross-run
    # noise. That contrast is the point: computing spread over the wrong
    # axis doesn't just give a different number, it can hide noise entirely.
    within_run_stdev = statistics.pstdev(runs[0].teams[t]["p_champion"] for t in score.TEAMS)
    ok &= check("within-run (wrong-axis) spread is exactly 0 for this fixture",
                within_run_stdev == 0.0)
    ok &= check("...while the correct across-run noise floor is the real, nonzero control value",
                floor["p_champion"] > 0 and abs(floor["p_champion"] - EXPECTED_JITTERED) < TOL)

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
