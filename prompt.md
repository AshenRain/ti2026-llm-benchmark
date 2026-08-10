# Benchmark prompt — TI 2026

Send exactly this, with the full contents of `context_pack.md` pasted where marked.
One fresh chat per run. Do not follow up, do not correct the model, do not ask it to fix its output. Save whatever it returns.

---

You are forecasting the group stage and final result of The International 2026, a Dota 2 tournament.

Below is a briefing document. You may also use anything else you know or can look up.

<briefing>
[PASTE THE FULL CONTENTS OF context_pack.md HERE]
</briefing>

## The group stage structure

16 teams play a Swiss stage. Every team finishes in exactly one of six outcome buckets, and the number of teams in each bucket is fixed by the format:

| Bucket | Meaning | Number of teams |
|---|---|---|
| `w4_l0` | 4 wins, 0 losses | exactly 1 |
| `w4_l1` | 4 wins, 1 loss | exactly 2 |
| `w4_l2` | 4 wins, 2 losses | exactly 5 |
| `w2_l4` | 2 wins, 4 losses | exactly 5 |
| `w1_l4` | 1 win, 4 losses | exactly 2 |
| `w0_l4` | 0 wins, 4 losses | exactly 1 |

The first three buckets advance to the main event (8 teams). The last three are eliminated (8 teams).

## Your task

For each of the 16 teams, give a probability distribution over those six buckets. Separately, give each team's probability of winning the whole tournament.

Your numbers must satisfy:

- For each team, the six bucket probabilities sum to 1.0
- Across all 16 teams, the probabilities in each bucket column sum to that bucket's team count: 1, 2, 5, 5, 2, 1
- Across all 16 teams, `p_champion` sums to 1.0
- For each team, `p_champion` is at most the sum of its three advancing buckets

Use the exact team names listed in the briefing.

## Output format

Return a single JSON object and nothing else. No markdown code fences, no explanation before or after, no trailing text. The response must start with `{` and end with `}`.

```
{
  "reasoning": "Your analysis, at most 300 words. Write this first.",
  "teams": [
    {
      "team": "Team Yandex",
      "swiss": {
        "w4_l0": 0.00,
        "w4_l1": 0.00,
        "w4_l2": 0.00,
        "w2_l4": 0.00,
        "w1_l4": 0.00,
        "w0_l4": 0.00
      },
      "p_champion": 0.00
    }
  ]
}
```

The `teams` array must contain all 16 teams. Use decimals between 0 and 1, not percentages. Do not round so aggressively that the constraints break.
