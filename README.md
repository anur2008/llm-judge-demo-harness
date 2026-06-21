# LLM-as-Judge Demo Harness

A small, self-contained harness that scores AI-generated answers against a
golden reference set using **Claude on Amazon Bedrock**. One dependency: `boto3`.

## Files
- `judge_harness.py` - the harness (reads the two JSON files, calls Bedrock, prints a scored table)
- `golden_qa_dataset_DEMO.json` - synthetic golden set (18 Q&A pairs, ground truth + key_facts)
- `answers_to_judge_DEMO.json` - candidate answers to score (strong / partial / wrong-but-fluent plants)
- `judge_results.json` - written after each run (for screenshots / slides)


## Setup
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


