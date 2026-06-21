# OrchestClaim — Damage Claim Verification System

Multi-modal evidence review system that verifies damage claims using images, claim conversations, user history, and evidence requirements. Built for the HackerRank Orchestrate hackathon.

## System Overview

For each claim in `dataset/claims.csv`, the system:

1. Extracts the damage claim from the conversation transcript
2. Inspects submitted images using GPT-4o-mini with vision
3. Checks image sufficiency against evidence requirements
4. Evaluates user history for risk context
5. Produces structured output: claim status, issue type, object part, severity, risk flags, and image-grounded justifications

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Set API key
echo "OPENAI_API_KEY=sk-..." > ../.env
```

## Usage

### Process claims.csv and produce output.csv

```bash
# Run from repo root
python main.py
```

Output is written to `output.csv` at the repo root.

### Run evaluation on sample claims

```bash
# Run from repo root
python evaluation/main.py
```

Evaluates against `dataset/sample_claims.csv` and prints per-field accuracy.

## Project Structure

```
code/                           # All source code lives here
├── main.py                     # Terminal entry point (process claims.csv)
├── claim_processor.py          # Single-claim pipeline with API call
├── prompt_builder.py           # Structured prompt construction
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── evaluation/
    ├── main.py                 # Evaluation on sample_claims.csv
    └── evaluation_report.md    # Operational analysis and accuracy metrics
```

## Output Schema

| Column | Description |
|---|---|
| `evidence_standard_met` | Whether image set is sufficient to evaluate |
| `evidence_standard_met_reason` | Short reason for evidence decision |
| `risk_flags` | Semicolon-separated risk flags or `none` |
| `issue_type` | Visible issue type (dent, scratch, crack, etc.) |
| `object_part` | Relevant object part (e.g. front_bumper, screen) |
| `claim_status` | `supported`, `contradicted`, or `not_enough_information` |
| `claim_status_justification` | Concise image-grounded explanation |
| `supporting_image_ids` | Image IDs supporting the decision |
| `valid_image` | Whether image set is usable for automated review |
| `severity` | `none`, `low`, `medium`, `high`, or `unknown` |

## Design Decisions

- **Model**: GPT-4o-mini for cost-effective vision+reasoning
- **Image handling**: Pillow converts all images to proper JPEG before encoding — many `.jpg` files in the dataset have incorrect file signatures (TIFF, PNG, WEBP headers)
- **Prompt**: Single-turn structured prompt with system rules, evidence requirements, user history, and base64-encoded images
- **Output**: JSON mode (`response_format: json_object`) for reliable structured parsing
- **Post-processing**: Rule-based corrections after API call — forces `contradicted` when model flags `wrong_object` or when part is visible but shows no damage
- **Retry**: Exponential backoff (3 attempts) on API errors; no retry on `400` invalid request errors
- **Throttle**: 1.5s between calls to stay under TPM limits
- **Risk flags**: Derived from image quality assessment, claim mismatch detection, and user history analysis