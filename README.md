# LLM-as-Judge Demo Harness

A small, self-contained harness that scores AI-generated answers against a
golden reference set using **Claude on Amazon Bedrock**. Built for a live
conference demo. One dependency: `boto3`.

## Files
- `judge_harness.py` - the harness (reads the two JSON files, calls Bedrock, prints a scored table)
- `golden_qa_dataset_DEMO.json` - synthetic golden set (18 Q&A pairs, ground truth + key_facts)
- `answers_to_judge_DEMO.json` - candidate answers to score (strong / partial / wrong-but-fluent plants)
- `judge_results.json` - written after each run (for screenshots / slides)

> All data is fictional ("Northwind Steelworks / Atlas ERP"). Nothing here is
> tied to any real product or customer. The `demo_note` / `expected_band`
> fields in the answers file are **speaker-only** and are never sent to the judge.

## Setup (do this before rehearsal)
```bash
pip install -r requirements.txt
```
Copy `.env.example` to `.env` (uses the same `int` AWS profile as
`automated-tests-v1` local runs), then load it:
```bash
cp .env.example .env    # first time only
set -a && source .env && set +a
```
The `int` profile is the one passed to `.robot.local.sh` in automated-tests-v1
(e.g. `./.robot.local.sh dev int int …`). Region is `us-west-2`, matching
`bedrockutils.py`.

Then confirm two things in the **Bedrock console**:
1. **Model access** is enabled for the Claude model you'll use.
2. The **model ID** and **region** match `BEDROCK_MODEL_ID` / `AWS_REGION` in
   `.env` (or the defaults at the top of `judge_harness.py`). Override per-run
   with `--model` / `--region` if needed.

## Verify your setup
Run these in order before rehearsal:

**(a) Offline dry-run** — no AWS call; confirms JSON loads and the table renders:
```bash
python judge_harness.py --dry-run --no-pause
```

**(b) Live smoke test** — scores one answer against Bedrock:
```bash
python judge_harness.py --limit 1 --no-pause
```

**(c) Full live demo** — all 18 answers, no pagination pauses:
```bash
python judge_harness.py --no-pause
```

Add `--rehearsal` to (b) or (c) to print judge-vs-`expected_band` calibration at the end.

## Run
```bash
# Full live run (18 answers, paginated 6 per page)
python judge_harness.py

# Record-friendly: no pauses
python judge_harness.py --no-pause

# Quick smoke test (first 4 only)
python judge_harness.py --limit 4

# Rehearsal: also print judge-vs-expected calibration at the end
python judge_harness.py --no-pause --rehearsal

# Offline render test (NO Bedrock call - rows marked SIMULATED; never use on stage)
python judge_harness.py --dry-run --no-pause
```

## Reading the output
Each answer prints:
- a **verdict** - PASS (4-5) · PARTIAL (3) · FAIL (1-2)
- an **overall** score out of 5
- the **per-criterion** scores (only the criteria in that question's `evaluation_focus`)
- a **one-line reason** from the judge

A closing summary shows the average, how many were flagged (≤2), and their IDs.

## The demo beats
- **Clean passes to open on:** Q03, Q04, Q09 (should score 5).
- **The money moment:** **Q01** - a fluent, confident answer claiming the
  project is healthy (78/100) when ground truth says Critical (46.5). Valid
  format, fluent prose, completely wrong - only ground-truth comparison catches it.
- **The subtle catch:** **Q06** - a plausible "trend is improving" answer that
  inverts the real (worsening) numbers.
- **Fact-level precision:** **Q14** - one substituted element (Manganese for
  Molybdenum) inside an otherwise-good answer.

## If a plant doesn't score as expected
The judge is itself non-deterministic - that's the point of the talk. If Q01
scores a 3 instead of a 1, tighten the rubric wording in `judge_harness.py`
(`RUBRIC` dict) or the judge instructions, and re-run. That *is* the calibration
story: you make the judge defensible before you trust it.

## Stage safety
Run live by default, but keep a screen recording of a clean rehearsal run as a
fallback. If the network or Bedrock misbehaves on the day, cut to the recording
without breaking stride.
