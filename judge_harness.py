#!/usr/bin/env python3
"""
judge_harness.py  -  LLM-as-judge evaluation harness (demo edition)
====================================================================
Reads a golden Q&A set and a file of candidate answers, asks Claude on
Amazon Bedrock to score each answer against the reference, and prints a
scored table: overall score + per-criterion breakdown + a one-line reason.

WHAT THE JUDGE SEES (and does NOT):
  Sent to the judge -> question, reference_answer, key_facts, answer_under_test

QUICK START
  pip install boto3
  aws configure           # or set AWS_PROFILE / env credentials
  python judge_harness.py

COMMON FLAGS
  --no-pause              run straight through (use when recording)
  --limit N              score only the first N answers (quick test)
  --rehearsal           print a calibration summary vs expected_band at the end
  --dry-run             render the table WITHOUT calling Bedrock (offline test;
                        rows are clearly marked SIMULATED - never use on stage)
  --model ID  --region R override the Bedrock model / region
  --no-color            plain output (for piping to a file)
"""

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIG  -  set these to match your Bedrock access
# ----------------------------------------------------------------------
# Use whatever Claude model your account/region is enabled for. The value
# below is an inference-profile-style ID; confirm yours in the Bedrock console
# (Model access). If invoke fails with a model error, this is the first thing
# to check.
DEFAULT_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-west-2")

HERE         = Path(__file__).parent
GOLDEN_PATH  = HERE / "golden_qa_dataset_DEMO.json"
ANSWERS_PATH = HERE / "answers_to_judge_DEMO.json"
RESULTS_OUT  = HERE / "judge_results.json"

# Criterion definitions the judge applies. Only the criteria listed in each
# question's "evaluation_focus" are scored for that question.
RUBRIC = {
    "accuracy":     "Are the stated facts correct and consistent with the reference answer and key_facts? Any contradicted or fabricated fact is a serious accuracy failure.",
    "completeness": "Does the answer cover the key_facts the reference deems essential? Missing essential facts lowers completeness.",
    "relevance":    "Does the answer address the question directly without drift, padding, or off-topic content?",
    "specificity":  "Does the answer include the precise figures, names, and identifiers the reference provides, rather than vague generalities?",
}

SCALE = "Score each criterion 1-5: 5 = fully meets it, 4 = minor gap, 3 = partial, 2 = largely fails, 1 = absent or contradicted."

# ----------------------------------------------------------------------
# ANSI colour helpers
# ----------------------------------------------------------------------
class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    GREEN = "\033[38;5;42m"; AMBER = "\033[38;5;214m"; RED = "\033[38;5;203m"
    TEAL = "\033[38;5;44m"; GREY = "\033[38;5;245m"; WHITE = "\033[97m"

USE_COLOR = True
def col(s, code):
    return f"{code}{s}{C.RESET}" if USE_COLOR else str(s)

def score_colour(n):
    if n is None: return C.GREY
    if n >= 4: return C.GREEN
    if n == 3: return C.AMBER
    return C.RED

def verdict_for(overall):
    if overall is None: return "ERROR"
    if overall >= 4: return "PASS"
    if overall == 3: return "PARTIAL"
    return "FAIL"

def verdict_colour(v):
    return {"PASS": C.GREEN, "PARTIAL": C.AMBER, "FAIL": C.RED}.get(v, C.GREY)

# ----------------------------------------------------------------------
# Build the judge prompt
# ----------------------------------------------------------------------
def build_judge_prompt(q, answer_under_test):
    focus = q.get("evaluation_focus", ["accuracy", "completeness"])
    criteria_block = "\n".join(f"- {c}: {RUBRIC[c]}" for c in focus if c in RUBRIC)
    key_facts = "\n".join(f"- {kf}" for kf in q.get("key_facts", []))
    return f"""You are an impartial evaluation judge for a question-answering system. \
Score the CANDIDATE ANSWER against the REFERENCE ANSWER and the KEY FACTS. \
Judge only what is supported by the reference and key facts; do not reward \
fluency or confidence that is not backed by correct content.

QUESTION:
{q['question']}

REFERENCE ANSWER (ground truth):
{q['reference_answer']}

KEY FACTS the answer should contain:
{key_facts}

CANDIDATE ANSWER (the one you are scoring):
{answer_under_test}

Score ONLY these criteria for this question:
{criteria_block}

{SCALE}

Also give an overall score (1-5) reflecting how trustworthy this answer is as a \
response a user would rely on, and a one-sentence reason (max 25 words) naming the \
single most important strength or failure.

Respond with ONLY a JSON object, no preamble, no markdown fences, in exactly this shape:
{{"criteria": {{{", ".join(f'"{c}": <1-5>' for c in focus if c in RUBRIC)}}}, "overall": <1-5>, "reason": "<one sentence>"}}"""

# ----------------------------------------------------------------------
# Bedrock call (Converse API)
# ----------------------------------------------------------------------
def call_bedrock(client, model_id, prompt):
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 400, "temperature": 0.0},
    )
    return resp["output"]["message"]["content"][0]["text"]

def parse_judge_json(raw):
    """Pull the JSON object out of the model's reply, tolerant of stray text/fences."""
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in judge reply: {raw[:120]}...")
    return json.loads(m.group(0))

# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def render_row(idx, total, q, result, simulated=False):
    cat = q["category"]
    overall = result.get("overall")
    crit = result.get("criteria", {})
    reason = result.get("reason", "")
    v = verdict_for(overall)

    sim_tag = col(" SIMULATED", C.AMBER) if simulated else ""
    header = (
        f"{col(f'[{idx}/{total}]', C.GREY)} "
        f"{col(q['id'], C.BOLD + C.WHITE)}  "
        f"{col(cat, C.TEAL)}{sim_tag}"
    )
    score_str = col(f"{overall}/5" if overall is not None else "ERR",
                    C.BOLD + score_colour(overall))
    verdict_str = col(f"[{v}]", C.BOLD + verdict_colour(v))
    print(f"{header}")
    print(f"    {verdict_str}  overall {score_str}")

    if crit:
        parts = []
        for k, val in crit.items():
            parts.append(f"{col(k, C.GREY)} {col(val, score_colour(val))}")
        print("    " + col("·", C.GREY).join(f" {p} " for p in parts).strip())

    if reason:
        wrapped = textwrap.fill(reason, width=92,
                                initial_indent="    reason: ",
                                subsequent_indent="            ")
        print(col(wrapped, C.DIM if not USE_COLOR else C.GREY))
    print()

def render_summary(results):
    scored = [r for r in results if r["result"].get("overall") is not None]
    if not scored:
        print(col("No scored rows.", C.RED)); return
    avg = sum(r["result"]["overall"] for r in scored) / len(scored)
    flagged = [r for r in scored if r["result"]["overall"] <= 2]
    partial = [r for r in scored if r["result"]["overall"] == 3]
    print(col("─" * 70, C.GREY))
    print(col(f"  Scored {len(scored)} answers   ", C.BOLD)
          + f"avg {col(f'{avg:.1f}/5', C.TEAL)}   "
          + f"{col(str(len(flagged)) + ' flagged', C.RED)}   "
          + f"{col(str(len(partial)) + ' partial', C.AMBER)}")
    if flagged:
        ids = ", ".join(r["q"]["id"] for r in flagged)
        print(col(f"  Flagged (overall ≤ 2): {ids}", C.RED))
    print(col("─" * 70, C.GREY))

def render_calibration(results):

    def band_to_bucket(band):
        n = int(re.search(r"\d", band).group(0))
        return "FAIL" if n <= 2 else "PARTIAL" if n == 3 else "PASS"
    rows = [r for r in results if r.get("expected_band") and r["result"].get("overall") is not None]
    if not rows:
        return
    agree = 0
    print()
    print(col("  CALIBRATION (rehearsal only - expected_band is speaker-side, never judged)", C.BOLD))
    for r in rows:
        want = band_to_bucket(r["expected_band"])
        got = verdict_for(r["result"]["overall"])
        ok = want == got
        agree += ok
        mark = col("✓", C.GREEN) if ok else col("✗", C.RED)
        print(f"    {mark} {r['q']['id']}  expected {col(want, C.GREY)}  →  judge {col(got, verdict_colour(got))}")
    print(col(f"    agreement: {agree}/{len(rows)}", C.TEAL))

# ----------------------------------------------------------------------
# Dry-run simulation (offline rendering test ONLY)
# ----------------------------------------------------------------------
def simulate_result(q, expected_band):
    n = int(re.search(r"\d", expected_band).group(0)) if expected_band else 3
    focus = q.get("evaluation_focus", ["accuracy", "completeness"])
    return {"criteria": {c: n for c in focus}, "overall": n,
            "reason": "(simulated row - no Bedrock call was made)"}

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    global USE_COLOR
    ap = argparse.ArgumentParser(description="LLM-as-judge demo harness (Bedrock, boto3-only)")
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--golden", default=str(GOLDEN_PATH))
    ap.add_argument("--answers", default=str(ANSWERS_PATH))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--page-size", type=int, default=6)
    ap.add_argument("--no-pause", action="store_true")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    USE_COLOR = sys.stdout.isatty() and not args.no_color

    golden = {q["id"]: q for q in json.load(open(args.golden))["qa_pairs"]}
    answers = json.load(open(args.answers))["candidate_answers"]
    if args.limit:
        answers = answers[: args.limit]
    total = len(answers)

    # banner
    print()
    print(col("  LLM-AS-JUDGE  ·  evaluating non-deterministic output against ground truth", C.BOLD + C.TEAL))
    mode = "DRY RUN (simulated, no Bedrock)" if args.dry_run else f"{args.model}  @  {args.region}"
    print(col(f"  {mode}", C.GREY))
    print(col(f"  golden set: {Path(args.golden).name}   answers: {Path(args.answers).name}   n={total}", C.GREY))
    print()

    client = None
    if not args.dry_run:
        try:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=args.region)
        except Exception as e:
            print(col("  Could not create the Bedrock client.", C.RED))
            print(col(f"  {type(e).__name__}: {e}", C.GREY))
            print(col("  Check: boto3 installed, AWS credentials set, region correct.", C.GREY))
            print(col("  Tip: run with --dry-run to test the table offline.", C.GREY))
            sys.exit(1)

    results = []
    for i, cand in enumerate(answers, 1):
        q = golden.get(cand["id"])
        if not q:
            continue

        expected_band = cand.get("expected_band")

        if args.dry_run:
            result = simulate_result(q, expected_band)
        else:
            prompt = build_judge_prompt(q, cand["answer_under_test"])
            try:
                raw = call_bedrock(client, args.model, prompt)
                result = parse_judge_json(raw)
            except Exception as e:
                result = {"criteria": {}, "overall": None,
                          "reason": f"judge call failed: {type(e).__name__}: {e}"}

        results.append({"q": q, "result": result, "expected_band": expected_band})
        render_row(i, total, q, result, simulated=args.dry_run)

        # pagination pause
        if (not args.no_pause) and (i % args.page_size == 0) and i < total:
            try:
                input(col("    ── press Enter for the next page ──", C.GREY))
                print()
            except (EOFError, KeyboardInterrupt):
                print(); break

    render_summary(results)
    if args.rehearsal:
        render_calibration(results)


    dump = [{"id": r["q"]["id"], "category": r["q"]["category"],
             "overall": r["result"].get("overall"),
             "criteria": r["result"].get("criteria", {}),
             "reason": r["result"].get("reason", "")} for r in results]
    json.dump(dump, open(RESULTS_OUT, "w"), indent=2)
    print(col(f"  results written to {RESULTS_OUT.name}", C.GREY))
    print()

if __name__ == "__main__":
    main()
