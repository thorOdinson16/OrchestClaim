# Evaluation Report

## Strategy

The system uses a single-turn structured prompt with GPT-4o-mini (vision) to evaluate each claim. The prompt includes:

1. **System prompt**: Role definition, allowed value enums, evidence requirements reference, and rules (ignore text instructions in images, ground justifications in visual evidence)
2. **User prompt**: Claim object + conversation transcript + user history + relevant evidence requirements + base64-encoded images

Output is constrained via `response_format: json_object` with `temperature: 0.1` for reproducibility.

## Model Calls

| Dataset | Rows | Images | Model Calls |
|---|---|---|---|
| Sample | 20 | ~25 | 20 |
| Test (claims.csv) | 44 | ~100 | 44 |
| **Total** | **64** | **~125** | **64** |

Each row requires exactly one model call. Retries (max 3 with exponential backoff) are not counted in the base numbers.

## Image Handling

Many `.jpg` files in the dataset have incorrect file signatures (actual formats include TIFF, PNG, RIFF/WEBP). The system uses Pillow to open and convert all images to proper JPEG before encoding, which eliminates `invalid_image_format` API errors.

## Token Usage (Measured via tiktoken)

| Component | Tokens per call |
|---|---|
| System prompt + evidence reqs | ~800 input tokens |
| Claim text + user history | ~300 input tokens |
| Per image (after Pillow conversion, detail=auto) | ~500-1,500 input tokens each |
| Output JSON | ~150 output tokens |

**Sample set**: ~25 images + 20 text prompts = ~47,000 input tokens; ~3,000 output tokens

**Test set**: ~100 images + 44 text prompts = ~148,400 input tokens; ~6,600 output tokens

**Total**: ~195,400 input tokens + ~9,600 output tokens

## Cost Estimate

Pricing (GPT-4o-mini):
- Input: $0.15 / 1M tokens
- Output: $0.60 / 1M tokens

| Dataset | Input Cost | Output Cost | Total |
|---|---|---|---|
| Sample | $0.007 | $0.002 | **$0.009** |
| Test | $0.022 | $0.004 | **$0.026** |
| **Total** | **$0.029** | **$0.006** | **$0.035** |

Full pipeline cost is under $0.04, making repeated runs economical.

## Latency

Average per-claim processing time: **~8.5 seconds** (measured over 44 test claims).

Total runtime for test set: **~7 minutes** (includes 1.5s throttle between calls).

## Rate Limit (TPM/RPM) Considerations

GPT-4o-mini standard tier: 10,000 RPM / 2,000,000 TPM.

- 44 claims at 1 call each with 1-4 images: some calls used ~200K TPM on multi-image claims
- A 1.5s delay between calls keeps usage under 120 RPM
- Initial runs hit TPM limits on 3-image claims; increased throttle to 1.5s resolved this
- Exponential backoff (up to 8s) handles residual rate-limit errors

## Batching, Caching, and Retry Strategy

- **Retry**: 3 retries with exponential backoff (2s, 4s, 8s) on API/parse errors
- **Retry — invalid request**: Immediate fail on `400 invalid_image_format` (not retried)
- **Throttle**: 1.5s sleep between calls to avoid TPM limits
- **Caching**: Not implemented; could hash (claim_text + image_paths) to skip re-processing
- **Batching**: Sequential (each call is independent); parallel asyncio could halve wall time

## Accuracy Metrics

Evaluated on 20 labeled sample claims.

| Field | Accuracy |
|---|---|
| Claim Status | 75.0% |
| Issue Type | 65.0% |
| Object Part | 80.0% |
| Evidence Standard Met | 80.0% |
| Valid Image | 80.0% |
| Severity | 45.0% |
| **Overall** | **70.8%** |

### Why claim_status is 75% (not higher)

- **Vision limitations** (3/5 errors): The model cannot resolve fine details in some images (e.g. subtle door dent, small scratch despite visible bumper). In these cases it outputs `not_enough_information` instead of `supported` or `contradicted`.
- **Hallucination** (1/5 errors): One case shows intact package seal but model hallucinates torn packaging.
- **Ambiguous labels** (1/5 errors): One sample claim labels two images of different cars as `not_enough_information`, but the model correctly flags it as `contradicted` — a reasonable disagreement.

### Severity accuracy (45%)

Severity estimation is inherently subjective. The model tends to overestimate (medium vs low) in about half of mismatched cases. A rule-based severity mapping from issue_type + object_part could improve this.

## Confusion Matrix (Claim Status)

| Expected \ Predicted | supported | contradicted | not_enough_information |
|---|---|---|---|
| supported | 11 | 0 | 1 |
| contradicted | 1 | 2 | 2 |
| not_enough_information | 0 | 1 | 2 |

## Key Observations

1. **Claim-mismatch detection**: The model correctly identifies when images show a different vehicle/object than claimed.
2. **Text-instruction risk**: The prompt explicitly instructs the model to ignore text instructions inside images (`text_instruction_present` flag).
3. **User history integration**: Users with >3 rejections or flagged history add `user_history_risk` and `manual_review_required` flags.
4. **Multilingual support**: GPT-4o-mini handles claims in English, Spanish, Hindi, and Chinese without additional configuration.
5. **Evidence standard**: The model evaluates whether images are sufficient based on the evidence requirements reference provided in the system prompt.

## Limitations

1. **No caching**: Each evaluation makes a fresh API call; no caching for repeated images or similar claims.
2. **Sequential processing**: Could be parallelized with asyncio for faster throughput.
3. **Single model**: Uses only GPT-4o-mini; a two-stage approach (detection → verification) could improve accuracy.
4. **No calibration**: Confidence scores from the model are not calibrated against human judgments.
