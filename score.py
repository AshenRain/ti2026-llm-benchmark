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
import random
import re
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

# These three strings are copied verbatim from context_pack.md's own team
# section headers (e.g. "**BoomBoys / BetBoom Team** -- Kiritych, ..."). A
# model that returns exactly this text followed prompt.md's "use the exact
# team names listed in the briefing" to the letter -- the briefing itself
# presents these particular teams under two textual forms (a compact
# section header and, separately, an explicit "X plays as Y" sentence), and
# prompt.md never enumerates a canonical list to disambiguate which one to
# use. That is a gap in our spec, not a model error, so resolving one of
# these must NOT count as a "repair" or downgrade parse_level -- see
# CLAUDE.md. Keep this table narrow: only the literal headers, not general
# shorthand (that's TEAM_SYNONYMS, which the model was never instructed to
# produce and so IS treated as a repair).
TEAM_SPEC_AMBIGUITY = {
    "boomboys / betboom team": "BoomBoys",
    "team vision / parivision": "Team Vision",
    "huligani (ex-l1ga team)": "HULIGANI",
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
      1 - valid JSON, every key and team name exactly as specified, OR
          resolved only via TEAM_SPEC_AMBIGUITY (see spec_ambiguities --
          that's a gap in our spec, not a format violation)
      2 - recovered mechanically (markdown fences / surrounding prose,
          key-name variants, TEAM_SYNONYMS-style team-name variants) or
          contains a team name that couldn't be resolved at all (see
          unknown_teams); usable, but a format violation -- see
          coherence_report()'s "format_violation" flag
      3 - not recoverable; excluded from scoring (ok=False)
    """
    path: Path
    model: str
    run_id: str
    ok: bool
    parse_level: int = 3
    parse_error: str | None = None
    repairs: list = field(default_factory=list)
    spec_ambiguities: list = field(default_factory=list)
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
    """Returns (resolved_name, kind). kind is one of:
      "exact"          -- already canonical, verbatim
      "case"           -- canonical name, differs only in case/spacing
      "spec_ambiguity" -- matches a context_pack.md team-section header
                          verbatim (TEAM_SPEC_AMBIGUITY); a gap in our
                          spec, not a model error -- must not affect
                          parse_level
      "synonym"        -- resolved via the general legacy/shorthand table
                          (TEAM_SYNONYMS); the model normalized on its own
                          initiative, which is a real repair
      "unresolved"     -- no match anywhere; resolved_name == raw, and it
                          will surface via unknown_teams
    """
    if raw in TEAMS:
        return raw, "exact"
    lowered = raw.strip().lower()
    for team in TEAMS:
        if team.lower() == lowered:
            return team, "case"
    if lowered in TEAM_SPEC_AMBIGUITY:
        return TEAM_SPEC_AMBIGUITY[lowered], "spec_ambiguity"
    if lowered in TEAM_SYNONYMS:
        return TEAM_SYNONYMS[lowered], "synonym"
    return raw, "unresolved"


def _process_team_entry(entry: dict, repairs: list, spec_ambiguities: list) -> tuple:
    """Returns (resolved_team_name_or_None, swiss_dict, p_champion)."""
    team_key, team_key_repair = _find_aliased_key(entry, TEAM_NAME_ALIASES, "team")
    if team_key is None or not isinstance(entry.get(team_key), str):
        return None, None, math.nan
    if team_key_repair:
        repairs.append(f"team-name key '{team_key}' resolved to 'team'")

    raw_name = entry[team_key]
    resolved_name, name_kind = _resolve_team_name(raw_name)
    if name_kind in ("case", "synonym"):
        repairs.append(f"team name '{raw_name}' resolved to '{resolved_name}'")
    elif name_kind == "spec_ambiguity":
        spec_ambiguities.append(
            f"team name '{raw_name}' resolved to '{resolved_name}' "
            "(matches a context_pack.md section header verbatim -- the briefing "
            "presents this team under two textual forms, prompt.md doesn't say which "
            "to use)")
    # "exact": nothing to log. "unresolved": nothing to log here either --
    # it surfaces via unknown_teams, which parse_run_file folds into parse_level.

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


_RUN_SUFFIX_RE = re.compile(r"^(?P<model>.+)_run_(?P<run_id>\d+)$")


def _split_model_run(stem: str) -> tuple:
    """Filenames are '{model}_run_{N}.json' (e.g. 'opus5_run_1' ->
    ('opus5', '1'), 'grok4_5_run_3' -> ('grok4_5', '3')). Falls back to a
    plain trailing '_{N}' split for anything that doesn't match, so
    filenames from before this convention still parse sensibly."""
    m = _RUN_SUFFIX_RE.match(stem)
    if m:
        return m.group("model"), m.group("run_id")
    if "_" in stem:
        model, run_id = stem.rsplit("_", 1)
        return model, run_id
    return stem, "?"


def parse_run_file(path: Path) -> ParsedRun:
    model, run_id = _split_model_run(path.stem)

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
    spec_ambiguities: list = []
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
        name, swiss, p_champion = _process_team_entry(entry, repairs, spec_ambiguities)
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

    # A team name that resolved via TEAM_SPEC_AMBIGUITY doesn't count against
    # parse_level (it's a spec gap, not a model error). But one that never
    # resolved at all -- still sitting in unknown_teams under its raw,
    # unrecognized name -- must: silently leaving it at level 1 would hide a
    # team that's effectively lost to every canonical-name lookup downstream
    # (ensemble, spread).
    parse_level = 2 if (repairs or unknown_teams) else 1
    return ParsedRun(path, model, run_id, ok=True, parse_level=parse_level, repairs=repairs,
                      spec_ambiguities=spec_ambiguities, teams=teams,
                      missing_teams=missing_teams, unknown_teams=unknown_teams)


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
        "spec_ambiguities": run.spec_ambiguities,
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
    for that team, and the contributing count is reported.

    Uses sample stdev (N-1), not population stdev (N): the model's runs are
    a small sample drawn to estimate the model's true noise, not the full
    population of interest -- see CLAUDE.md. With N=3, population stdev
    understates this by a factor of sqrt(3/2) (~1.225x), which would
    overstate the significance of any between-model gap compared against
    it."""
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
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
        result[team] = {
            "p_champion": {
                "n": len(champ_vals),
                "stdev": statistics.stdev(champ_vals) if len(champ_vals) > 1 else 0.0,
            },
            "swiss": bucket_spread,
        }
    return result


def noise_floor(spread: dict) -> dict:
    """Reduces spread_across_runs()'s per-team breakdown to the model-level
    noise floor CLAUDE.md calls for: for p_champion and each of the six
    Swiss buckets, the mean -- across the 16 teams -- of that team's own
    stdev across the model's runs. This is NOT the spread of the 16 teams'
    values within a single run (that would answer a different question:
    how much teams differ from each other, not how noisy repeat-running
    the same model is)."""
    champ_stdevs = [v["p_champion"]["stdev"] for v in spread.values() if v["p_champion"]["n"] > 1]
    result = {"p_champion": statistics.fmean(champ_stdevs) if champ_stdevs else None}
    for bucket in BUCKETS:
        vals = [v["swiss"][bucket]["stdev"] for v in spread.values() if v["swiss"][bucket]["n"] > 1]
        result[bucket] = statistics.fmean(vals) if vals else None
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


def _load_bookmaker_implied(odds_path: Path) -> dict:
    """source -> {team: raw implied probability (1/decimal_odds)}, BEFORE
    normalization. Raw implied probabilities are an intermediate value only
    (see CLAUDE.md) -- their column sums to more than 1.0 by design (the
    platform's overround/margin lives in that excess). Never report these
    numbers as a final probability; use bookmaker_baselines() for that."""
    by_source: dict[str, dict[str, float]] = defaultdict(dict)
    with odds_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            odds_str = (row.get("decimal_odds") or "").strip()
            if not odds_str:
                continue
            by_source[row["bookmaker"]][row["team"]] = 1.0 / float(odds_str)
    return by_source


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
    baselines = {}
    for source, implied in _load_bookmaker_implied(odds_path).items():
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
# Scale diagnostics
# ---------------------------------------------------------------------------
#
# Every probability column in this project is supposed to be normalized
# (rows/columns sum to a fixed target -- see coherence checks above and
# CLAUDE.md's Swiss-format constraints). Raw values -- most concretely,
# 1/decimal_odds before dividing out a bookmaker's overround -- are an
# intermediate step only and must never be reported or compared against
# normalized numbers. This section makes every column's sum visible in one
# place and turns "a column claimed to be normalized isn't" into a loud,
# explicit signal instead of a silent scale mismatch.
#
# A failed check is not automatically "an error" in the same sense, though.
# Two very different things can make a normalized column miss its target:
#
#   "pipeline" -- a column WE build deterministically (the uniform baseline,
#       a bookmaker source's post-normalization values) doesn't sum right,
#       or a raw value ended up compared as if it were normalized. Either
#       way that is a defect in this code and needs fixing before any
#       number in the report can be trusted.
#   "measured" -- a model run's own swiss/champion column (or the ensemble,
#       which is a plain mean of model runs and so inherits whatever they
#       didn't satisfy) misses its target. That is the benchmark's actual
#       finding -- the model failed to follow the constraints -- not a bug.

PIPELINE_CATEGORY = "pipeline"
MEASURED_CATEGORY = "measured"


def _scale_check(label: str, values, is_normalized: bool, expected: float | None = None,
                  category: str | None = None, tol: float = TOLERANCE) -> dict:
    clean = [v for v in values if not math.isnan(v)]
    total = sum(clean)
    error = is_normalized and expected is not None and abs(total - expected) > tol
    return {
        "label": label,
        "sum": total,
        "n": len(clean),
        "is_normalized": is_normalized,
        "expected": expected,
        "category": category if error else None,
        "error": error,
    }


def scale_diagnostics(runs: list[ParsedRun], odds_path: Path) -> list:
    """One record per probability column across every entrant this project
    reports on: each usable model run's six Swiss buckets and champion
    column (category=measured), the uniform baseline's same columns
    (category=pipeline), the ensemble baseline's same columns
    (category=measured -- it's a mean of model runs and inherits their
    incoherence), and each bookmaker source's champion column in both its
    raw (pre-normalization, never checked) and normalized (category=
    pipeline) form. Normalized columns are checked against their expected
    total and flagged with error=True if they miss it by more than `tol`."""
    checks = []

    for run in runs:
        if not run.ok:
            continue
        prefix = f"model {run.model} run {run.run_id}"
        for bucket in BUCKETS:
            vals = [info["swiss"][bucket] for info in run.teams.values()]
            checks.append(_scale_check(f"{prefix}: swiss[{bucket}]", vals, True,
                                        BUCKET_OCCUPANCY[bucket], MEASURED_CATEGORY))
        champ_vals = [info["p_champion"] for info in run.teams.values()]
        checks.append(_scale_check(f"{prefix}: p_champion", champ_vals, True, 1.0, MEASURED_CATEGORY))

    uniform = uniform_baseline()
    for bucket in BUCKETS:
        vals = [info["swiss"][bucket] for info in uniform.values()]
        checks.append(_scale_check(f"baseline uniform: swiss[{bucket}]", vals, True,
                                    BUCKET_OCCUPANCY[bucket], PIPELINE_CATEGORY))
    champ_vals = [info["p_champion"] for info in uniform.values()]
    checks.append(_scale_check("baseline uniform: p_champion", champ_vals, True, 1.0, PIPELINE_CATEGORY))

    ensemble = ensemble_baseline(runs)
    if ensemble:
        for bucket in BUCKETS:
            vals = [info["swiss"][bucket] for info in ensemble.values()]
            checks.append(_scale_check(f"baseline ensemble: swiss[{bucket}]", vals, True,
                                        BUCKET_OCCUPANCY[bucket], MEASURED_CATEGORY))
        champ_vals = [info["p_champion"] for info in ensemble.values()]
        checks.append(_scale_check("baseline ensemble: p_champion", champ_vals, True, 1.0, MEASURED_CATEGORY))

    for source, implied in _load_bookmaker_implied(odds_path).items():
        checks.append(_scale_check(f"bookmaker {source}: p_champion (raw, pre-normalization)",
                                    implied.values(), False))
    for source, teams in bookmaker_baselines(odds_path).items():
        champ_vals = [info["p_champion"] for info in teams.values()]
        checks.append(_scale_check(f"bookmaker {source}: p_champion (normalized)", champ_vals, True, 1.0,
                                    PIPELINE_CATEGORY))

    return checks


def _print_scale_diagnostics(checks: list) -> bool:
    """Prints the column-sum table, one line per column. A failed check is
    labeled by category, not lumped into one generic "SCALE ERROR":
      [PIPELINE ERROR]        -- a defect in this code (see PIPELINE_CATEGORY
                                  in scale_diagnostics()); needs fixing
      [MEASURED INCOHERENCE]  -- a model (or the ensemble) missed a
                                  constraint; that's the benchmark's finding,
                                  not a bug
    Returns True only for a pipeline error -- callers use this (and only
    this) to decide a non-zero exit code. Measured incoherence must never
    fail the script; it's expected, reportable data."""
    print("\n--- scale diagnostics (column sums) ---")
    pipeline_error = False
    measured_incoherence = False
    for c in checks:
        tag = "normalized" if c["is_normalized"] else "raw"
        line = f"  [{tag:10s}] {c['label']}: sum={c['sum']:.6f} (n={c['n']})"
        if c["expected"] is not None:
            line += f"  expected={c['expected']:.6f}"
        if c["error"]:
            if c["category"] == PIPELINE_CATEGORY:
                line += "  *** PIPELINE ERROR: code defect, needs fixing ***"
                pipeline_error = True
            else:
                line += "  *** MEASURED INCOHERENCE: constraint violated in the data ***"
                measured_incoherence = True
        print(line)
    if pipeline_error:
        print("\n  PIPELINE ERROR: a column this code builds deterministically (the uniform")
        print("  baseline, a bookmaker source's normalization) did not sum correctly, or a")
        print("  raw value was compared as if normalized. This is a code defect -- do not")
        print("  trust any number in this report until it's fixed.")
    if measured_incoherence:
        print("\n  Measured incoherence: one or more model runs (or the ensemble, which")
        print("  inherits it) violated a row/column/champion constraint. This is the")
        print("  benchmark's actual finding, not a tooling error.")
    return pipeline_error
    return any_error


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
# Charts
# ---------------------------------------------------------------------------
#
# matplotlib is the one non-stdlib dependency in this project (see
# requirements.txt), imported lazily here so `coherence`/`score` keep
# working without it installed. Colors are the validated categorical
# palette from the dataviz skill (references/palette.md): this specific
# 5-of-8 ordering (blue, yellow, magenta, green, violet) is the one that
# clears the all-pairs CVD and normal-vision floors for a scatter-style
# chart -- `node scripts/validate_palette.js "<hexes>" --mode light
# --pairs all` confirms it (worst CVD ΔE 13.0, worst normal-vision ΔE
# 16.3). Model identity is doubly encoded (color + marker shape) so it
# never rests on color alone.

MODEL_ORDER = ["opus5", "gpt_sol5_6", "grok4_5", "deepseek_chat_search-on", "gemini_flash_lite"]
MODEL_COLORS = {
    "opus5": "#2a78d6",
    "gpt_sol5_6": "#eda100",
    "grok4_5": "#e87ba4",
    "deepseek_chat_search-on": "#008300",
    "gemini_flash_lite": "#4a3aa7",
}
MODEL_MARKERS = {
    "opus5": "o",
    "gpt_sol5_6": "s",
    "grok4_5": "^",
    "deepseek_chat_search-on": "D",
    "gemini_flash_lite": "P",
}

CHART_SURFACE = "#fcfcfb"
CHART_INK_PRIMARY = "#0b0b0b"
CHART_INK_SECONDARY = "#52514e"
CHART_GRIDLINE = "#e1e0d9"
CHART_TARGET_COLOR = "#3a3a38"


def plot_column_sums(runs: list[ParsedRun], output_path: Path) -> None:
    """One point per (model, run, bucket): the six Swiss-bucket column
    sums, against a horizontal target mark at each bucket's fixed
    occupancy (1, 2, 5, 5, 2, 1). Three points per model per bucket (one
    per run); models are dodged apart within each bucket and a small fixed
    -seed jitter keeps near-identical repeat runs from fully overlapping,
    without touching the y-value (the actual data). Unrecoverable (level
    3) runs contribute nothing, matching every other report in this file.

    Deterministic and self-contained: run this function (or `py score.py
    charts`) against the same runs/ and it reproduces the same PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok_runs = [r for r in runs if r.ok]
    by_model = group_by_model(ok_runs)
    models_present = [m for m in MODEL_ORDER if m in by_model]
    models_present += sorted(m for m in by_model if m not in MODEL_ORDER)

    n_models = max(len(models_present), 1)
    dodge = [(-0.32 + 0.64 * i / (n_models - 1)) if n_models > 1 else 0.0 for i in range(n_models)]
    rng = random.Random(0)

    fig, ax = plt.subplots(figsize=(8.2, 9.6), dpi=200)
    fig.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    for bi, bucket in enumerate(BUCKETS):
        target = BUCKET_OCCUPANCY[bucket]
        ax.hlines(target, bi - 0.4, bi + 0.4, colors=CHART_TARGET_COLOR, linewidth=3.5, zorder=2)

    for model, offset in zip(models_present, dodge):
        color = MODEL_COLORS.get(model, CHART_INK_SECONDARY)
        marker = MODEL_MARKERS.get(model, "o")
        xs, ys = [], []
        for bi, bucket in enumerate(BUCKETS):
            for run in by_model[model]:
                vals = [info["swiss"][bucket] for info in run.teams.values() if not math.isnan(info["swiss"][bucket])]
                xs.append(bi + offset + (rng.random() - 0.5) * 0.06)
                ys.append(sum(vals))
        ax.scatter(xs, ys, s=130, color=color, marker=marker, edgecolors=CHART_INK_PRIMARY,
                   linewidths=0.9, alpha=0.88, label=model, zorder=3)

    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels(BUCKETS, fontsize=16)
    ax.set_xlim(-0.6, len(BUCKETS) - 0.4)
    ax.set_xlabel("Swiss bucket", fontsize=18, color=CHART_INK_PRIMARY, labelpad=12)
    ax.set_ylabel("column sum", fontsize=18, color=CHART_INK_PRIMARY, labelpad=12)
    ax.set_title("Swiss bucket column sums vs. target occupancy", fontsize=20,
                 color=CHART_INK_PRIMARY, pad=18, fontweight="bold", wrap=True)

    ax.tick_params(axis="both", labelsize=15, colors=CHART_INK_SECONDARY)
    ax.grid(axis="y", color=CHART_GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CHART_GRIDLINE)

    handles, labels = ax.get_legend_handles_labels()
    target_handle = plt.Line2D([], [], color=CHART_TARGET_COLOR, linewidth=3.5)
    handles.append(target_handle)
    labels.append("target")
    ax.legend(handles, labels, fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, frameon=False, labelcolor=CHART_INK_PRIMARY, handletextpad=0.6,
              columnspacing=1.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


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
        if report["spec_ambiguities"]:
            print("  spec ambiguities (name resolved; does not affect parse_level):")
            for note in _summarize_repairs(report["spec_ambiguities"]):
                print(f"    - {note}")
        if report["missing_teams"]:
            print(f"  missing teams: {report['missing_teams']}")
        if report["unknown_teams"]:
            print(f"  unknown teams (unresolved -- this pushed parse_level to 2): {report['unknown_teams']}")
        print(f"  row deviation (max): {report['row_deviation_max']:.4f}")
        print(f"  column deviation (max): {report['column_deviation_max']:.4f}")
        print(f"  champion deviation: {report['champion_deviation']:.4f}")
        if report["advance_violations"]:
            print(f"  advance violations: {report['advance_violations']}")

    print("\n--- parse summary, by model ---")
    for model, s in parse_summary(runs).items():
        print(f"  {model}: n={s['n_usable']}/{s['n_total']} parsed "
              f"(level1={s['n_level1']}, level2={s['n_level2']}, unrecoverable={s['n_level3']})")

    all_spec_ambiguities = sorted({note for run in runs for note in run.spec_ambiguities})
    if all_spec_ambiguities:
        print("\n--- spec ambiguities across all runs (for README disclosure) ---")
        for note in all_spec_ambiguities:
            print(f"  - {note}")

    print("\n--- spread across runs (noise floor), by model ---")
    print("    for each team: sample stdev (N-1) of its value across the model's runs; then mean over the 16 teams")
    for model, model_runs in group_by_model(runs).items():
        if len(model_runs) < 2:
            continue
        floor = noise_floor(spread_across_runs(model_runs))
        print(f"\n  {model} ({len(model_runs)} runs):")
        if floor["p_champion"] is not None:
            print(f"    p_champion: {floor['p_champion']:.4f}")
        for bucket in BUCKETS:
            if floor[bucket] is not None:
                print(f"    swiss[{bucket}]: {floor[bucket]:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["coherence", "score", "charts"])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--results", type=Path, default=Path("results.csv"))
    parser.add_argument("--odds", type=Path, default=Path("odds.csv"))
    parser.add_argument("--charts-dir", type=Path, default=Path("charts"))
    parser.add_argument("--tol", type=float, default=TOLERANCE)
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"No run files found in {args.runs_dir}/")
        return 0

    if args.command == "charts":
        out = args.charts_dir / "column_sums.png"
        plot_column_sums(runs, out)
        print(f"wrote {out}")
        return 0

    if args.command == "coherence":
        _print_coherence(runs, args.tol)
        pipeline_error = _print_scale_diagnostics(scale_diagnostics(runs, args.odds))
        return 1 if pipeline_error else 0

    results = load_results(args.results)
    if not results:
        print(f"{args.results} has no filled-in outcomes yet — nothing to score.")
        return 0
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

    pipeline_error = _print_scale_diagnostics(scale_diagnostics(runs, args.odds))
    return 1 if pipeline_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
