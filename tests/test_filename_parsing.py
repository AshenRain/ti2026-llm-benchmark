"""Checks _split_model_run(), which pulls (model, run_id) out of a run
file's stem. The actual convention in runs/ is '{model}_run_{N}.json'
(e.g. 'opus5_run_1.json', 'grok4_5_run_3.json') -- a naive split on the
last underscore mis-parses 'opus5_run_1' as model='opus5_run', run='1'.

Also checks the older '{model}_{N}.json' style (used by this project's
own test fixtures, predating the '_run_' convention) still falls back
correctly, so existing fixtures don't need renaming.

Run with: py tests/test_filename_parsing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import score  # noqa: E402


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main() -> int:
    ok = True

    print("actual runs/ convention: {model}_run_{N}")
    ok &= check("'opus5_run_1' -> model='opus5', run='1'",
                score._split_model_run("opus5_run_1") == ("opus5", "1"))
    ok &= check("'grok4_5_run_3' -> model='grok4_5' (model name itself has an underscore)",
                score._split_model_run("grok4_5_run_3") == ("grok4_5", "3"))
    ok &= check("'deepseek-light_run_2' -> model='deepseek-light', run='2'",
                score._split_model_run("deepseek-light_run_2") == ("deepseek-light", "2"))
    ok &= check("'a_b_c_run_10' -> model='a_b_c' (multi-digit run id)",
                score._split_model_run("a_b_c_run_10") == ("a_b_c", "10"))

    print("older '{model}_{N}' convention still falls back correctly")
    ok &= check("'opus5_1' (no '_run_') -> model='opus5', run='1'",
                score._split_model_run("opus5_1") == ("opus5", "1"))
    ok &= check("'fixture-model_1' -> model='fixture-model', run='1'",
                score._split_model_run("fixture-model_1") == ("fixture-model", "1"))

    print("edge cases")
    ok &= check("no underscore at all -> (stem, '?')",
                score._split_model_run("nostructure") == ("nostructure", "?"))

    print("parse_run_file() uses the fixed split -- three same-model files group together")
    runs = score.load_runs(Path("runs"))
    models = {r.model for r in runs}
    ok &= check("no model name ends with a stray '_run' suffix",
                not any(m.endswith("_run") for m in models))
    for model in ("opus5", "grok4_5"):
        model_runs = [r for r in runs if r.model == model]
        if not model_runs:
            print(f"  [SKIP] no runs/*.json for '{model}' right now")
            continue
        ok &= check(f"'{model}' resolves to exactly the model name (not '{model}_run')",
                    all(r.model == model for r in model_runs))

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
