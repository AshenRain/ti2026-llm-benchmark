# TI 2026 LLM Prediction Benchmark

## What this is

A pre-registered benchmark of how well frontier LLMs forecast a Dota 2 tournament, and — more importantly — whether their probability estimates are internally coherent and well calibrated.

Deliverables are two LinkedIn posts: one before the tournament (methodology and coherence results), one after (calibration and accuracy).

## Timeline

- **2026-08-10** — context pack frozen, prompt finalised
- **2026-08-11** — all model runs collected
- **2026-08-12** — odds snapshot, coherence analysis, post #1 published, repo pushed
- **2026-08-13 04:00 CEST** — predictions lock in the Dota 2 client. Nothing may change after this
- **2026-08-13 to 08-23** — tournament
- **2026-08-24/25** — scoring, charts, post #2

The push before the lock is the pre-registration. Do not amend `context_pack.md`, `prompt.md`, or anything in `runs/` after that point.

## Design decisions already made

These were settled deliberately. Do not reopen them without asking.

- **No betting odds in the context pack.** Odds are an independent baseline in the evaluation. Feeding them to the models would let them copy the answer.
- **No analyst power rankings in the pack.** Same reason — they encode someone else's forecast.
- **The TaiLung ban and Topson signing stay in the pack.** They broke on 9–10 August. Including them puts models with and without web access on equal footing.
- **Bucket constraints are stated explicitly in the prompt.** Valve publishes them in the client, so hiding them would make the task ambiguous. This turns coherence into an instruction-following test with an unambiguous pass/fail.
- **Three runs per model, each in a fresh chat.** Chat interfaces give no temperature control; a new chat is how we get an independent sample. Runs from ChatGPT were produced by a third party on their own account — settings there are not controlled, and this is a stated limitation.
- **Web search is allowed for every model.** DeepSeek is run twice: strong model without search, light model with search. That contrast is a secondary result, not a confound to eliminate.
- **Multiple odds sources in `odds.csv` are scored separately, never merged.** Each value in the `bookmaker` column (e.g. `esportbet_aggregate`, `polymarket`) is normalised within its own group and scored as its own baseline entrant, alongside the models, uniform, and ensemble. Overround differs a lot between sources — averaging before normalising would let the highest-margin source distort the rest, and averaging after normalising would quietly erase disagreement between sources that is itself worth reporting.
- **Bookmaker/market baselines are champion-only.** Odds sources price the outright winner, not the six Swiss buckets, so they have no bucket-level distribution. They participate only in champion-level metrics (champion Brier, champion log loss); bucket-level metrics (multiclass Brier/log loss, advance Brier) are `N/A` for them by design, not 0. The uniform baseline is the only baseline scored on bucket-level metrics alongside the models and the ensemble. `score.py` must report this `N/A` explicitly — never substitute a default value or drop the row silently.
- **The within-model noise floor uses sample stdev (N-1), not population stdev (N).** For each team, `spread_across_runs()` takes the stdev of that team's value across a model's three runs, then averages over the 16 teams — the three runs are a *sample* used to estimate the model's true noise, not the entire population of interest, so Bessel's correction applies. Population stdev understates this by a factor of `sqrt(3/2)` (~1.225x) at N=3, and since this number is the yardstick every between-model gap gets compared against, that understatement would systematically inflate how significant those gaps look. `tests/fixtures/spread_check/` is a control fixture with a known, hand-derived expected value for exactly this formula.

## Swiss format

16 teams, six terminal buckets with fixed occupancy:

| Bucket | Record | Teams | Outcome |
|---|---|---|---|
| `w4_l0` | 4–0 | 1 | advances |
| `w4_l1` | 4–1 | 2 | advances |
| `w4_l2` | 4–2 | 5 | advances |
| `w2_l4` | 2–4 | 5 | eliminated |
| `w1_l4` | 1–4 | 2 | eliminated |
| `w0_l4` | 0–4 | 1 | eliminated |

Each model returns a 16×6 matrix plus a champion column. Rows sum to 1. Columns sum to 1, 2, 5, 5, 2, 1. Champion column sums to 1.

## What to compute

**Before the tournament (coherence — needs no results):**
- Row sum deviation from 1.0, per model per run
- Column sum deviation from the fixed occupancy vector
- Champion column sum deviation from 1.0
- Violations of `p_champion <= p(advance)`
- Spread across the three runs of the same model, as the noise floor

**After the tournament:**
- Multiclass Brier score and log loss over the six buckets
- Binary Brier for advancing (top 8) and for champion
- Reliability diagram, predictions binned by 0.1
- Between-model differences plotted against the within-model spread. If the gap sits inside the noise floor, say so plainly — that is the honest finding, not a failure

**Baselines, all mandatory:**
- Bookmaker odds from `odds.csv`, converted with 1/odds and normalised to remove overround. Each source in the `bookmaker` column is its own entrant (see Design decisions) and is champion-only — `N/A` on bucket-level metrics
- Uniform: 0.5 to advance, 1/16 to win
- Ensemble: mean across all models, scored as its own entrant

## Repo layout

```
context_pack.md    frozen model input
prompt.md          exact prompt sent, with paste marker
runs/              {model}_run_{run}.json, raw and unedited
odds.csv           bookmaker snapshot, 2026-08-12
results.csv        actual outcomes, filled after 08-23
score.py           scoring and charts
CLAUDE.md          this file
README.md          public writeup
```

## House rules

- Raw model output in `runs/` is never hand-edited. If a model returns malformed JSON, that is data — record the failure and parse defensively in `score.py`.
- Every number in either post must be reproducible from `score.py`.
- Sample size is small. Nothing in the writeup may claim one model beats another unless the gap exceeds the within-model spread.
- All reports and charts use normalized probabilities. Raw values (e.g. `1/decimal_odds` before dividing out a bookmaker's overround) are an intermediate step only and are never surfaced as a final number outside `score.py`'s scale-diagnostics table — mixing the two scales (as happened once with Team Yandex's raw 0.167/0.230 vs. normalized 0.109/0.196) must never pass unnoticed. `score.py`'s scale diagnostics print every probability column's sum and explicitly error if a column marked normalized doesn't sum to its expected total within `1e-6`.
