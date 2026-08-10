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
import statistics
from collections import Counter, defaultdict
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
BUCKET_RECORDS = {"w4_l0": (4, 0), "w4_l1": (4, 1), "w4_l2": (4, 2),
                   "w2_l4": (2, 4), "w1_l4": (1, 4), "w0_l4": (0, 4)}

TOLERANCE = 1e-6  # what counts as "exactly satisfies the constraint"

# Legacy/alternate org tags noted in context_pack.md section 2, plus common
# shorthand a model might use instead of the full roster name. Matched
# case-insensitively. Resolving one of these is a mechanical repair (level
# 2), not a free pass -- the prompt asks for exact names on purpose.
TEAM_SYNONYMS = {
    "betboom team": "BoomBoys",
    "betboom": "BoomBoys",
    "bb team": "BoomBoys",
    "parivision": "Team Vision",
    "pvision": "Team Vision",
    "tundra esports": "1win Team",
    "tundra": "1win Team",
    "l1ga team": "HULIGANI",
    "l1ga": "HULIGANI",
    "yandex": "Team Yandex",
    "falcons": "Team Falcons",
    "liquid": "Team Liquid",
    "1win": "1win Team",
    "xtreme": "Xtreme Gaming",
    "aurora": "Aurora Gaming",
    "spirit": "Team Spirit",
    "nigma": "Nigma Galaxy",
    "vici": "Vici Gaming",
    "resilience": "Team Resilience",
    "gamer legion": "GamerLegion",
    "lgd": "LGD Gaming",
}

# Key names a model might use instead of the ones specified in prompt.md.
# Resolving any of these (or even just a case/spacing variant of the
# canonical key) is a mechanical repair.
TEAMS_LIST_ALIASES = {"teams", "predictions", "results", "team_predictions", "forecasts"}
TEAM_NAME_ALIASES = {"team", "name", "team_name", "teamname"}
SWISS_ALIASES = {"swiss", "buckets", "bucket_probs", "distribution", "probabilities",
                  "swiss_probabilities", "swiss_distribution"}
CHAMPION_ALIASES = {"p_champion", "champion_probability", "win_probability", "p_win",
                     "champion", "champion_prob", "pchampion"}


# ---------------------------------------------------------------------------
# Loading and defensive parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedRun:
    """parse_level:
      1 - valid JSON, every key and team name exactly as specified
      2 - recovered mechanically (markdown fences / surrounding prose,
          key-name variants, team-name variants); usable, but a format
          violation -- see coherence_report()'s "format_violation" flag
      3 - not recoverable; excluded from scoring (ok=False)
    """
    path: Path
    model: str
    run_id: str
    ok: bool
    parse_level: int = 3
    parse_error: str | None = None
    repairs: list = field(default_factory=list)
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


def _normalize_key(key: str) -> str:
    """Collapse a key to lowercase alphanumerics so 'Team_Name', 'team name'
    and 'TeamName' all compare equal."""
    return "".join(ch for ch in key.strip().lower() if ch.isalnum())


def _find_aliased_key(d: dict, aliases: set, canonical: str) -> tuple:
    """Look for `canonical`, or a case/spacing variant of it, or one of
    `aliases`, among d's keys. Returns (actual_key_found_or_None,
    was_a_repair). Exact-canonical match is not a repair; anything else
    (including a same-spelling-different-case canonical key) is."""
    if canonical in d:
        return canonical, False
    normalized_map = {_normalize_key(k): k for k in d.keys()}
    norm_canonical = _normalize_key(canonical)
    if norm_canonical in normalized_map:
        return normalized_map[norm_canonical], True
    for alias in aliases:
        norm_alias = _normalize_key(alias)
        if norm_alias in normalized_map:
            return normalized_map[norm_alias], True
    return None, False


def _resolve_bucket_key(raw_key: str) -> tuple:
    """Match a Swiss bucket key that may use different separators/case, or
    plain win-loss notation ('4-0', '40'). Returns (canonical_bucket_or_None,
    was_a_repair)."""
    if raw_key in BUCKETS:
        return raw_key, False
    compact = _normalize_key(raw_key)
    for bucket, (wins, losses) in BUCKET_RECORDS.items():
        if compact == f"w{wins}l{losses}" or compact == f"{wins}{losses}":
            return bucket, True
    return None, False


def _resolve_team_name(raw: str) -> tuple:
    """Returns (resolved_name, was_a_repair). If unresolved, resolved_name
    is the raw string as given (it will surface via unknown_teams)."""
    if raw in TEAMS:
        return raw, False
    lowered = raw.strip().lower()
    for team in TEAMS:
        if team.lower() == lowered:
            return team, True
    if lowered in TEAM_SYNONYMS:
        return TEAM_SYNONYMS[lowered], True
    return raw, False


def _process_team_entry(entry: dict, repairs: list) -> tuple:
    """Returns (resolved_team_name_or_None, swiss_dict, p_champion)."""
    team_key, team_key_repair = _find_aliased_key(entry, TEAM_NAME_ALIASES, "team")
    if team_key is None or not isinstance(entry.get(team_key), str):
        return None, None, math.nan
    if team_key_repair:
        repairs.append(f"team-name key '{team_key}' resolved to 'team'")

    raw_name = entry[team_key]
    resolved_name, name_repair = _resolve_team_name(raw_name)
    if name_repair:
        repairs.append(f"team name '{raw_name}' resolved to '{resolved_name}'")

    swiss_key, swiss_key_repair = _find_aliased_key(entry, SWISS_ALIASES, "swiss")
    swiss_raw = entry.get(swiss_key) if swiss_key else None
    if not isinstance(swiss_raw, dict):
        swiss_raw = {}
    elif swiss_key_repair:
        repairs.append(f"swiss key '{swiss_key}' (team {resolved_name}) resolved to 'swiss'")

    swiss = {b: math.nan for b in BUCKETS}
    for raw_bucket_key, val in swiss_raw.items():
        bucket, bucket_repair = _resolve_bucket_key(raw_bucket_key)
        if bucket is None or not isinstance(val, (int, float)):
            continue
        swiss[bucket] = float(val)
        if bucket_repair:
            repairs.append(f"bucket key '{raw_bucket_key}' (team {resolved_name}) resolved to '{bucket}'")

    champ_key, champ_key_repair = _find_aliased_key(entry, CHAMPION_ALIASES, "p_champion")
    p_champion_raw = entry.get(champ_key) if champ_key else None
    p_champion = float(p_champion_raw) if isinstance(p_champion_raw, (int, float)) else math.nan
    if champ_key is not None and champ_key_repair:
        repairs.append(f"champion key '{champ_key}' (team {resolved_name}) resolved to 'p_champion'")

    return resolved_name, swiss, p_champion


def parse_run_file(path: Path) -> ParsedRun:
    stem = path.stem
    if "_" in stem:
        model, run_id = stem.rsplit("_", 1)
    else:
        model, run_id = stem, "?"

    raw = path.read_text(encoding="utf-8")
    extraction_repair = False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(_extract_json_object(raw))
            extraction_repair = True
        except (ValueError, json.JSONDecodeError) as exc:
            return ParsedRun(path, model, run_id, ok=False, parse_level=3, parse_error=str(exc))

    if not isinstance(data, dict):
        return ParsedRun(path, model, run_id, ok=False, parse_level=3,
                          parse_error="top-level JSON is not an object")

    repairs: list = []
    if extraction_repair:
        repairs.append("extracted JSON object from surrounding text (markdown fences and/or prose)")

    teams_key, teams_key_repair = _find_aliased_key(data, TEAMS_LIST_ALIASES, "teams")
    teams_field = data.get(teams_key) if teams_key else None
    if not isinstance(teams_field, list):
        return ParsedRun(path, model, run_id, ok=False, parse_level=3,
                          parse_error="no usable 'teams' field found, even after key-alias lookup")
    if teams_key_repair:
        repairs.append(f"top-level key '{teams_key}' resolved to 'teams'")

    teams: dict[str, dict] = {}
    for entry in teams_field:
        if not isinstance(entry, dict):
            continue
        name, swiss, p_champion = _process_team_entry(entry, repairs)
        if name is None:
            continue
        teams[name] = {"swiss": swiss, "p_champion": p_champion}

    if not teams:
        return ParsedRun(path, model, run_id, ok=False, parse_level=3,
                          parse_error="'teams' field present but no usable team entries in it")

    known = set(TEAMS)
    seen = set(teams.keys())
    missing_teams = sorted(known - seen)
    unknown_teams = sorted(seen - known)

    parse_level = 2 if repairs else 1
    return ParsedRun(path, model, run_id, ok=True, parse_level=parse_level, repairs=repairs,
                      teams=teams, missing_teams=missing_teams, unknown_teams=unknown_teams)


def load_runs(runs_dir: Path) -> list[ParsedRun]:
    return [parse_run_file(p) for p in sorted(runs_dir.glob("*.json"))]


def parse_summary(runs: list[ParsedRun]) -> dict:
    """Per-model counts of how many runs parsed at each level -- the 'n
    parsed runs per model' the report must always surface."""
    summary = {}
    for model, model_runs in group_by_model(runs).items():
        counts = {1: 0, 2: 0, 3: 0}
        for r in model_runs:
            counts[r.parse_level] += 1
        summary[model] = {
            "n_total": len(model_runs),
            "n_level1": counts[1],
            "n_level2": counts[2],
            "n_level3": counts[3],
            "n_usable": counts[1] + counts[2],
        }
    return summary


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
                "ok": False, "parse_level": run.parse_level, "parse_error": run.parse_error}

    row_dev = row_deviations(run)
    col_dev = column_deviations(run)
    return {
        "path": str(run.path),
        "model": run.model,
        "run_id": run.run_id,
        "ok": True,
        "parse_level": run.parse_level,
        "format_violation": run.parse_level == 2,
        "repairs": run.repairs,
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


def bookmaker_baselines(odds_path: Path) -> dict:
    """Champion probabilities from odds.csv, grouped by the `bookmaker`
    column and normalized independently within each group (1/decimal_odds,
    divided by that group's own total to remove its own overround).

    Sources are kept separate rather than merged into one number: they can
    carry very different overround, and disagreement between them is a
    finding, not noise to average away (see CLAUDE.md).

    Returns {source: {team: {"swiss": None, "p_champion": float}}}. A
    source's "swiss" is always None -- bookmaker/market odds only price the
    outright champion market, so these baselines are N/A for bucket-level
    metrics (multiclass Brier/log loss, advance Brier) and participate only
    in champion-level scoring. Returns {} if odds.csv has no populated rows
    yet."""
    by_source: dict[str, dict[str, float]] = defaultdict(dict)
    with odds_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            odds_str = (row.get("decimal_odds") or "").strip()
            if not odds_str:
                continue
            by_source[row["bookmaker"]][row["team"]] = 1.0 / float(odds_str)

    baselines = {}
    for source, implied in by_source.items():
        total = sum(implied.values())
        baselines[source] = {
            team: {"swiss": None, "p_champion": p / total}
            for team, p in implied.items()
        }
    return baselines


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


def multiclass_brier(predictions: dict, results: dict):
    """Mean multiclass Brier score over the six Swiss buckets. Returns None
    (not 0, not silently omitted) if this entrant has no bucket-level data
    at all -- e.g. a bookmaker/market baseline that only prices champion."""
    scores = []
    for team, bucket in results.items():
        info = predictions.get(team)
        if info is None or info.get("swiss") is None:
            continue
        probs = info["swiss"]
        if any(math.isnan(v) for v in probs.values()):
            continue
        scores.append(sum((probs[b] - (1.0 if b == bucket else 0.0)) ** 2 for b in BUCKETS))
    return statistics.fmean(scores) if scores else None


def multiclass_logloss(predictions: dict, results: dict, eps: float = 1e-15):
    scores = []
    for team, bucket in results.items():
        info = predictions.get(team)
        if info is None or info.get("swiss") is None:
            continue
        p_raw = info["swiss"].get(bucket)
        if p_raw is None or math.isnan(p_raw):
            continue
        p = min(max(p_raw, eps), 1 - eps)
        scores.append(-math.log(p))
    return statistics.fmean(scores) if scores else None


def binary_advance_brier(predictions: dict, results: dict):
    """Returns None for entrants with no bucket-level data (see
    multiclass_brier) -- advancing is derived from the swiss buckets."""
    scores = []
    for team, bucket in results.items():
        info = predictions.get(team)
        if info is None or info.get("swiss") is None:
            continue
        swiss = info["swiss"]
        if any(math.isnan(swiss[b]) for b in ADVANCE_BUCKETS):
            continue
        p_advance = sum(swiss[b] for b in ADVANCE_BUCKETS)
        actual = 1.0 if bucket in ADVANCE_BUCKETS else 0.0
        scores.append((p_advance - actual) ** 2)
    return statistics.fmean(scores) if scores else None


def champion_brier(predictions: dict, champion: str):
    if champion is None:
        return None
    scores = []
    for team, info in predictions.items():
        p = info.get("p_champion")
        if p is None or math.isnan(p):
            continue
        actual = 1.0 if team == champion else 0.0
        scores.append((p - actual) ** 2)
    return statistics.fmean(scores) if scores else None


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

def _summarize_repairs(repairs: list) -> list:
    """Collapse repairs that repeat per-team (e.g. the same key alias
    applied to all 16 entries) into one line with a count, for a readable
    report. The full, per-team detail stays available on ParsedRun.repairs."""
    templates = []
    for r in repairs:
        start = r.find(" (team ")
        end = r.find(")", start) if start != -1 else -1
        templates.append(r[:start] + r[end + 1:] if start != -1 and end != -1 else r)
    counts = Counter(templates)
    out, seen = [], set()
    for template in templates:
        if template in seen:
            continue
        seen.add(template)
        n = counts[template]
        out.append(f"{template} (x{n})" if n > 1 else template)
    return out


def _print_coherence(runs: list[ParsedRun], tol: float) -> None:
    for run in runs:
        report = coherence_report(run, tol)
        level_tag = f"parse level {report['parse_level']}"
        if report.get("format_violation"):
            level_tag += " [FORMAT VIOLATION]"
        print(f"\n{report['path']}  (model={report['model']}, run={report['run_id']}, {level_tag})")
        if not report["ok"]:
            print(f"  UNRECOVERABLE: {report['parse_error']}")
            continue
        if report["repairs"]:
            print("  repairs applied:")
            for repair in _summarize_repairs(report["repairs"]):
                print(f"    - {repair}")
        if report["missing_teams"]:
            print(f"  missing teams: {report['missing_teams']}")
        if report["unknown_teams"]:
            print(f"  unknown teams: {report['unknown_teams']}")
        print(f"  row deviation (max): {report['row_deviation_max']:.4f}")
        print(f"  column deviation (max): {report['column_deviation_max']:.4f}")
        print(f"  champion deviation: {report['champion_deviation']:.4f}")
        if report["advance_violations"]:
            print(f"  advance violations: {report['advance_violations']}")

    print("\n--- parse summary, by model ---")
    for model, s in parse_summary(runs).items():
        print(f"  {model}: n={s['n_usable']}/{s['n_total']} parsed "
              f"(level1={s['n_level1']}, level2={s['n_level2']}, unrecoverable={s['n_level3']})")

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

    def fmt(x):
        return "N/A" if x is None else f"{x:.4f}"

    print("--- parse summary, by model ---")
    for model, s in parse_summary(runs).items():
        print(f"  {model}: n={s['n_usable']}/{s['n_total']} parsed "
              f"(level1={s['n_level1']}, level2={s['n_level2']}, unrecoverable={s['n_level3']})")

    print("\n--- per-run metrics (level-2 runs are format violations; still scored) ---")
    for run in runs:
        if not run.ok:
            continue
        tag = " [FORMAT VIOLATION]" if run.parse_level == 2 else ""
        print(f"\n{run.path}{tag}")
        print(f"  multiclass Brier: {fmt(multiclass_brier(run.teams, results))}")
        print(f"  multiclass log loss: {fmt(multiclass_logloss(run.teams, results))}")
        print(f"  advance Brier: {fmt(binary_advance_brier(run.teams, results))}")
        print(f"  champion Brier: {fmt(champion_brier(run.teams, champion))}")

    metric_fns = {
        "multiclass Brier": lambda r: multiclass_brier(r.teams, results),
        "multiclass log loss": lambda r: multiclass_logloss(r.teams, results),
        "advance Brier": lambda r: binary_advance_brier(r.teams, results),
        "champion Brier": lambda r: champion_brier(r.teams, champion),
    }

    def mean_or_none(run_subset, fn):
        vals = [v for v in (fn(r) for r in run_subset) if v is not None]
        return statistics.fmean(vals) if vals else None

    print("\n--- per model: strict (level 1 only) vs all (level 1+2) ---")
    for model, model_runs in group_by_model(runs).items():
        level1_runs = [r for r in model_runs if r.parse_level == 1]
        usable_runs = [r for r in model_runs if r.ok]
        if not usable_runs:
            continue
        print(f"\n{model}  (n_level1={len(level1_runs)}, n_level1+2={len(usable_runs)})")
        for label, fn in metric_fns.items():
            strict = mean_or_none(level1_runs, fn)
            lenient = mean_or_none(usable_runs, fn)
            print(f"  {label}: strict(L1)={fmt(strict)}  all(L1+L2)={fmt(lenient)}")

    print("\n--- baselines ---")
    entrants = {"uniform": uniform_baseline(), "ensemble": ensemble_baseline(runs)}
    for source, teams in bookmaker_baselines(args.odds).items():
        entrants[f"bookmaker ({source})"] = teams

    for label, teams in entrants.items():
        print(f"\n{label}")
        print(f"  multiclass Brier: {fmt(multiclass_brier(teams, results))}")
        print(f"  multiclass log loss: {fmt(multiclass_logloss(teams, results))}")
        print(f"  advance Brier: {fmt(binary_advance_brier(teams, results))}")
        print(f"  champion Brier: {fmt(champion_brier(teams, champion))}")


if __name__ == "__main__":
    main()
