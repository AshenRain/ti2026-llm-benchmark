"""Scoring and coherence checks for the TI 2026 LLM prediction benchmark.

Two phases, per CLAUDE.md:
  - coherence: runs before the tournament, needs only runs/*.json
  - score:     runs after the tournament, additionally needs results.csv

Raw files in runs/ are never hand-edited. Malformed model output is data;
this module parses it defensively and records what went wrong instead of
raising.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

TEAMS = [
    "Team Yandex",
    "BoomBoys",
    "Team Falcons",
    "Team Liquid",
    "1win Team",
    "Xtreme Gaming",
    "Aurora Gaming",
    "Team Vision",
    "Team Spirit",
    "Nigma Galaxy",
    "HULIGANI",
    "Vici Gaming",
    "Team Resilience",
    "OG",
    "GamerLegion",
    "LGD Gaming",
]

BUCKETS = ["w4_l0", "w4_l1", "w4_l2", "w2_l4", "w1_l4", "w0_l4"]
BUCKET_OCCUPANCY = {"w4_l0": 1, "w4_l1": 2, "w4_l2": 5, "w2_l4": 5, "w1_l4": 2, "w0_l4": 1}
ADVANCE_BUCKETS = ["w4_l0", "w4_l1", "w4_l2"]
ELIMINATE_BUCKETS = ["w2_l4", "w1_l4", "w0_l4"]

TOLERANCE = 1e-6  # what counts as "exactly satisfies the constraint"


# ---------------------------------------------------------------------------
# Loading and defensive parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedRun:
    path: Path
    model: str
    run_id: str
    ok: bool
    parse_error: str | None = None
    teams: dict = field(default_factory=dict)   # team -> {"swiss": {bucket: prob}, "p_champion": float}
    missing_teams: list = field(default_factory=list)
    unknown_teams: list = field(default_factory=list)


def _extract_json_object(text: str) -> str:
    """Best-effort recovery of a JSON object from text that isn't pure JSON
    (markdown fences, leading/trailing prose), per the house rule that
    malformed output is data, not something to discard."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in text")
    return text[start:end + 1]


def parse_run_file(path: Path) -> ParsedRun:
    stem = path.stem
    if "_" in stem:
        model, run_id = stem.rsplit("_", 1)
    else:
        model, run_id = stem, "?"

    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(_extract_json_object(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            return ParsedRun(path, model, run_id, ok=False, parse_error=str(exc))

    teams_field = data.get("teams") if isinstance(data, dict) else None
    if not isinstance(teams_field, list):
        return ParsedRun(path, model, run_id, ok=False, parse_error="'teams' field missing or not a list")

    teams: dict[str, dict] = {}
    for entry in teams_field:
        if not isinstance(entry, dict) or "team" not in entry:
            continue
        name = entry["team"]
        swiss_raw = entry.get("swiss", {}) if isinstance(entry.get("swiss"), dict) else {}
        swiss = {}
        for bucket in BUCKETS:
            val = swiss_raw.get(bucket)
            swiss[bucket] = float(val) if isinstance(val, (int, float)) else math.nan
        p_champion = entry.get("p_champion")
        teams[name] = {
            "swiss": swiss,
            "p_champion": float(p_champion) if isinstance(p_champion, (int, float)) else math.nan,
        }

    known = set(TEAMS)
    seen = set(teams.keys())
    missing_teams = sorted(known - seen)
    unknown_teams = sorted(seen - known)

    return ParsedRun(path, model, run_id, ok=True, teams=teams,
                      missing_teams=missing_teams, unknown_teams=unknown_teams)


def load_runs(runs_dir: Path) -> list[ParsedRun]:
    return [parse_run_file(p) for p in sorted(runs_dir.glob("*.json"))]


# ---------------------------------------------------------------------------
# Coherence checks (pre-tournament)
# ---------------------------------------------------------------------------

def row_deviations(run: ParsedRun) -> dict:
    """Per team: |sum of six bucket probs - 1.0|. NaN for unparseable rows."""
    out = {}
    for team, info in run.teams.items():
        vals = info["swiss"].values()
        if any(math.isnan(v) for v in vals):
            out[team] = math.nan
        else:
            out[team] = abs(sum(vals) - 1.0)
    return out


def column_deviations(run: ParsedRun) -> dict:
    """Per bucket: |sum over teams - fixed occupancy|."""
    out = {}
    for bucket in BUCKETS:
        vals = [info["swiss"][bucket] for info in run.teams.values()]
        vals = [v for v in vals if not math.isnan(v)]
        out[bucket] = abs(sum(vals) - BUCKET_OCCUPANCY[bucket])
    return out


def champion_deviation(run: ParsedRun) -> float:
    vals = [info["p_champion"] for info in run.teams.values() if not math.isnan(info["p_champion"])]
    return abs(sum(vals) - 1.0)


def advance_violations(run: ParsedRun, tol: float = TOLERANCE) -> list:
    """Teams where p_champion exceeds the sum of the three advancing buckets."""
    violations = []
    for team, info in run.teams.items():
        p_champ = info["p_champion"]
        advance = sum(info["swiss"][b] for b in ADVANCE_BUCKETS)
        if math.isnan(p_champ) or math.isnan(advance):
            continue
        if p_champ > advance + tol:
            violations.append((team, p_champ, advance))
    return violations


def coherence_report(run: ParsedRun, tol: float = TOLERANCE) -> dict:
    if not run.ok:
        return {"path": str(run.path), "model": run.model, "run_id": run.run_id,
                "ok": False, "parse_error": run.parse_error}

    row_dev = row_deviations(run)
    col_dev = column_deviations(run)
    return {
        "path": str(run.path),
        "model": run.model,
        "run_id": run.run_id,
        "ok": True,
        "missing_teams": run.missing_teams,
        "unknown_teams": run.unknown_teams,
        "row_deviation_max": max(row_dev.values(), default=math.nan),
        "row_deviations": row_dev,
        "column_deviation_max": max(col_dev.values(), default=math.nan),
        "column_deviations": col_dev,
        "champion_deviation": champion_deviation(run),
        "advance_violations": advance_violations(run, tol),
    }


def spread_across_runs(runs: list[ParsedRun]) -> dict:
    """Noise floor: stdev of each team's p_champion and per-bucket probs
    across runs of the same model. Runs where a team is absent are skipped
    for that team, and the contributing count is reported."""
    ok_runs = [r for r in runs if r.ok]
    result = {}
    for team in TEAMS:
        champ_vals = [r.teams[team]["p_champion"] for r in ok_runs
                      if team in r.teams and not math.isnan(r.teams[team]["p_champion"])]
        bucket_spread = {}
        for bucket in BUCKETS:
            vals = [r.teams[team]["swiss"][bucket] for r in ok_runs
                    if team in r.teams and not math.isnan(r.teams[team]["swiss"][bucket])]
            bucket_spread[bucket] = {
                "n": len(vals),
                "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            }
        result[team] = {
            "p_champion": {
                "n": len(champ_vals),
                "stdev": statistics.pstdev(champ_vals) if len(champ_vals) > 1 else 0.0,
            },
            "swiss": bucket_spread,
        }
    return result


def group_by_model(runs: list[ParsedRun]) -> dict:
    groups: dict[str, list[ParsedRun]] = {}
    for r in runs:
        groups.setdefault(r.model, []).append(r)
    return groups


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def uniform_baseline() -> dict:
    """0.5 to advance, 1/16 to win, distributed consistently with occupancy."""
    swiss = {b: BUCKET_OCCUPANCY[b] / len(TEAMS) for b in BUCKETS}
    return {team: {"swiss": dict(swiss), "p_champion": 1 / len(TEAMS)} for team in TEAMS}


def bookmaker_baseline(odds_path: Path) -> dict:
    """Champion probabilities from odds.csv: 1/decimal_odds, normalized to
    remove overround. Returns {} if odds.csv has no populated rows yet."""
    raw = {}
    with odds_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            odds_str = (row.get("decimal_odds") or "").strip()
            if not odds_str:
                continue
            raw[row["team"]] = 1.0 / float(odds_str)
    if not raw:
        return {}
    total = sum(raw.values())
    return {team: implied / total for team, implied in raw.items()}


def ensemble_baseline(runs: list[ParsedRun]) -> dict:
    """Mean across models: average each model's runs first, then average
    across models, so a model run three times doesn't dominate the mean."""
    groups = group_by_model(runs)
    per_model_means = []
    for model_runs in groups.values():
        ok_runs = [r for r in model_runs if r.ok]
        if not ok_runs:
            continue
        model_mean = {}
        for team in TEAMS:
            entries = [r.teams[team] for r in ok_runs if team in r.teams]
            if not entries:
                continue
            swiss = {b: statistics.fmean(e["swiss"][b] for e in entries if not math.isnan(e["swiss"][b]))
                      for b in BUCKETS}
            champ_vals = [e["p_champion"] for e in entries if not math.isnan(e["p_champion"])]
            model_mean[team] = {"swiss": swiss, "p_champion": statistics.fmean(champ_vals) if champ_vals else math.nan}
        per_model_means.append(model_mean)

    ensemble = {}
    for team in TEAMS:
        entries = [m[team] for m in per_model_means if team in m]
        if not entries:
            continue
        swiss = {b: statistics.fmean(e["swiss"][b] for e in entries) for b in BUCKETS}
        ensemble[team] = {"swiss": swiss, "p_champion": statistics.fmean(e["p_champion"] for e in entries)}
    return ensemble


# ---------------------------------------------------------------------------
# Scoring (post-tournament)
# ---------------------------------------------------------------------------

def load_results(results_path: Path) -> dict:
    """team -> actual swiss_bucket. Teams with an empty bucket are skipped
    (results.csv not filled in yet)."""
    results = {}
    with results_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bucket = (row.get("swiss_bucket") or "").strip()
            if bucket:
                results[row["team"]] = bucket
    return results


def load_champion(results_path: Path) -> str | None:
    with results_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            placement = (row.get("final_placement") or "").strip()
            if placement == "1":
                return row["team"]
    return None


def multiclass_brier(predictions: dict, results: dict) -> float:
    scores = []
    for team, bucket in results.items():
        if team not in predictions:
            continue
        probs = predictions[team]["swiss"]
        scores.append(sum((probs[b] - (1.0 if b == bucket else 0.0)) ** 2 for b in BUCKETS))
    return statistics.fmean(scores) if scores else math.nan


def multiclass_logloss(predictions: dict, results: dict, eps: float = 1e-15) -> float:
    scores = []
    for team, bucket in results.items():
        if team not in predictions:
            continue
        p = min(max(predictions[team]["swiss"][bucket], eps), 1 - eps)
        scores.append(-math.log(p))
    return statistics.fmean(scores) if scores else math.nan


def binary_advance_brier(predictions: dict, results: dict) -> float:
    scores = []
    for team, bucket in results.items():
        if team not in predictions:
            continue
        p_advance = sum(predictions[team]["swiss"][b] for b in ADVANCE_BUCKETS)
        actual = 1.0 if bucket in ADVANCE_BUCKETS else 0.0
        scores.append((p_advance - actual) ** 2)
    return statistics.fmean(scores) if scores else math.nan


def champion_brier(predictions: dict, champion: str) -> float:
    if champion is None:
        return math.nan
    scores = []
    for team, info in predictions.items():
        actual = 1.0 if team == champion else 0.0
        scores.append((info["p_champion"] - actual) ** 2)
    return statistics.fmean(scores) if scores else math.nan


def reliability_bins(pairs: list, bin_size: float = 0.1) -> list:
    """pairs: list of (predicted_prob, actual_outcome in {0,1}).
    Returns per-bin (lo, hi, mean_predicted, mean_actual, n)."""
    n_bins = int(round(1.0 / bin_size))
    bins = [[] for _ in range(n_bins)]
    for p, y in pairs:
        idx = min(int(p / bin_size), n_bins - 1)
        bins[idx].append((p, y))
    report = []
    for i, bucket in enumerate(bins):
        lo, hi = i * bin_size, (i + 1) * bin_size
        if bucket:
            mean_pred = statistics.fmean(p for p, _ in bucket)
            mean_actual = statistics.fmean(y for _, y in bucket)
            report.append((lo, hi, mean_pred, mean_actual, len(bucket)))
        else:
            report.append((lo, hi, math.nan, math.nan, 0))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_coherence(runs: list[ParsedRun], tol: float) -> None:
    for run in runs:
        report = coherence_report(run, tol)
        print(f"\n{report['path']}  (model={report['model']}, run={report['run_id']})")
        if not report["ok"]:
            print(f"  PARSE FAILED: {report['parse_error']}")
            continue
        if report["missing_teams"]:
            print(f"  missing teams: {report['missing_teams']}")
        if report["unknown_teams"]:
            print(f"  unknown teams: {report['unknown_teams']}")
        print(f"  row deviation (max): {report['row_deviation_max']:.4f}")
        print(f"  column deviation (max): {report['column_deviation_max']:.4f}")
        print(f"  champion deviation: {report['champion_deviation']:.4f}")
        if report["advance_violations"]:
            print(f"  advance violations: {report['advance_violations']}")

    print("\n--- spread across runs (noise floor), by model ---")
    for model, model_runs in group_by_model(runs).items():
        if len(model_runs) < 2:
            continue
        spread = spread_across_runs(model_runs)
        champ_stdevs = [v["p_champion"]["stdev"] for v in spread.values() if v["p_champion"]["n"] > 1]
        if champ_stdevs:
            print(f"  {model}: mean stdev(p_champion) across {len(model_runs)} runs = "
                  f"{statistics.fmean(champ_stdevs):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["coherence", "score"])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--results", type=Path, default=Path("results.csv"))
    parser.add_argument("--odds", type=Path, default=Path("odds.csv"))
    parser.add_argument("--tol", type=float, default=TOLERANCE)
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"No run files found in {args.runs_dir}/")
        return

    if args.command == "coherence":
        _print_coherence(runs, args.tol)
        return

    results = load_results(args.results)
    if not results:
        print(f"{args.results} has no filled-in outcomes yet — nothing to score.")
        return
    champion = load_champion(args.results)

    for run in runs:
        if not run.ok:
            continue
        print(f"\n{run.path}")
        print(f"  multiclass Brier: {multiclass_brier(run.teams, results):.4f}")
        print(f"  multiclass log loss: {multiclass_logloss(run.teams, results):.4f}")
        print(f"  advance Brier: {binary_advance_brier(run.teams, results):.4f}")
        print(f"  champion Brier: {champion_brier(run.teams, champion):.4f}")


if __name__ == "__main__":
    main()
