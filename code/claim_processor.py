import json
import os
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from prompt_builder import build_system_prompt, build_user_prompt

load_dotenv()

SYSTEM_PROMPT_CACHE: dict[str, str] = {}
_EVIDENCE_REQS_CACHE: list[dict] | None = None

ALLOWED = {
    "issue_type": [
        "dent", "scratch", "crack", "glass_shatter", "broken_part",
        "missing_part", "torn_packaging", "crushed_packaging",
        "water_damage", "stain", "none", "unknown",
    ],
    "claim_status": ["supported", "contradicted", "not_enough_information"],
    "severity": ["none", "low", "medium", "high", "unknown"],
    "object_part_car": [
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender",
        "quarter_panel", "body", "unknown",
    ],
    "object_part_laptop": [
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base", "body", "unknown",
    ],
    "object_part_package": [
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown",
    ],
    "risk_flags": [
        "none", "blurry_image", "cropped_or_obstructed",
        "low_light_or_glare", "wrong_angle", "wrong_object",
        "wrong_object_part", "damage_not_visible", "claim_mismatch",
        "possible_manipulation", "non_original_image",
        "text_instruction_present", "user_history_risk",
        "manual_review_required",
    ],
}


def load_evidence_reqs_text(csv_path: str) -> str:
    import pandas as pd

    df = pd.read_csv(csv_path)
    lines = []
    for _, r in df.iterrows():
        lines.append(
            f"- {r['requirement_id']}: object={r['claim_object']}, "
            f"applies_to={r['applies_to']}, "
            f"requirement={r['minimum_image_evidence']}"
        )
    return "\n".join(lines)


def get_object_parts(claim_object: str) -> list[str]:
    key = f"object_part_{claim_object}"
    return ALLOWED.get(key, ALLOWED["object_part_car"])


def validate_field(value: str, allowed: list[str], default: str = "unknown") -> str:
    if value in allowed:
        return value
    low = value.lower().strip()
    for a in allowed:
        if a == low:
            return a
    return default


def validate_risk_flags(value: str) -> str:
    if not value or value.strip().lower() == "none":
        return "none"
    flags = [f.strip().lower() for f in value.split(";")]
    valid = []
    for f in flags:
        if f in ALLOWED["risk_flags"]:
            valid.append(f)
    return ";".join(valid) if valid else "none"


def post_process_claim_status(pred: dict, user_history_row: dict | None = None) -> dict:
    """Apply rule-based corrections based on output fields and user history."""
    risk_set = set(
        f.strip()
        for f in pred.get("risk_flags", "").split(";")
        if f.strip() and f.strip() != "none"
    )

    if "wrong_object" in risk_set or "claim_mismatch" in risk_set:
        if pred.get("claim_status") != "contradicted":
            pred["claim_status"] = "contradicted"

    issue_type = pred.get("issue_type", "unknown")
    object_part = pred.get("object_part", "unknown")
    ev_std_met = pred.get("evidence_standard_met", False)

    if issue_type == "none" and object_part != "unknown":
        if pred.get("claim_status") == "supported":
            pred["claim_status"] = "contradicted"
            risk_set.add("claim_mismatch")

    if object_part == "unknown" and issue_type in ("unknown", "none") and not ev_std_met:
        if pred.get("claim_status") == "supported":
            pred["claim_status"] = "not_enough_information"

    if ev_std_met and object_part == "unknown" and issue_type == "unknown":
        pred["evidence_standard_met"] = False

    if user_history_row:
        hist_flags = str(user_history_row.get("history_flags", "")).strip()
        if "user_history_risk" in hist_flags:
            risk_set.add("user_history_risk")
        if "manual_review_required" in hist_flags:
            risk_set.add("manual_review_required")
        rej = int(user_history_row.get("rejected_claim", 0) or 0)
        if rej >= 3:
            risk_set.add("user_history_risk")
            risk_set.add("manual_review_required")

    pred["risk_flags"] = validate_risk_flags(";".join(sorted(risk_set))) if risk_set else "none"
    return pred


def process_claim(
    row: dict,
    user_history_df,
    evidence_reqs_df,
    evidence_reqs_text: str,
    dataset_dir: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> dict:
    claim_obj = str(row.get("claim_object", "")).strip().lower()
    user_claim = str(row.get("user_claim", ""))
    image_paths_str = str(row.get("image_paths", ""))
    image_paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]
    user_id = str(row.get("user_id", ""))

    user_history_row = None
    if user_history_df is not None:
        match = user_history_df[user_history_df["user_id"] == user_id]
        if not match.empty:
            user_history_row = match.iloc[0].to_dict()

    filtered_reqs = []
    if evidence_reqs_df is not None:
        mask = (evidence_reqs_df["claim_object"] == claim_obj) | (
            evidence_reqs_df["claim_object"] == "all"
        )
        filtered_reqs = evidence_reqs_df[mask].to_dict("records")

    content_parts, image_ids = build_user_prompt(
        claim_obj=claim_obj,
        user_claim=user_claim,
        user_history_row=user_history_row,
        evidence_reqs=filtered_reqs,
        image_paths=image_paths,
        dataset_dir=dataset_dir,
    )

    system_prompt = build_system_prompt(evidence_reqs_text)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_parts},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw)

            result = {
                "evidence_standard_met": bool(data.get("evidence_standard_met", False)),
                "evidence_standard_met_reason": str(data.get("evidence_standard_met_reason", "")),
                "risk_flags": validate_risk_flags(data.get("risk_flags", "none")),
                "issue_type": validate_field(
                    data.get("issue_type", "unknown"), ALLOWED["issue_type"]
                ),
                "object_part": validate_field(
                    data.get("object_part", "unknown"), get_object_parts(claim_obj)
                ),
                "claim_status": validate_field(
                    data.get("claim_status", "not_enough_information"),
                    ALLOWED["claim_status"],
                    default="not_enough_information",
                ),
                "claim_status_justification": str(data.get("claim_status_justification", "")),
                "supporting_image_ids": str(data.get("supporting_image_ids", "none")),
                "valid_image": bool(data.get("valid_image", False)),
                "severity": validate_field(
                    data.get("severity", "unknown"), ALLOWED["severity"]
                ),
            }
            result = post_process_claim_status(result, user_history_row)
            return result

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "evidence_standard_met": False,
                "evidence_standard_met_reason": f"JSON parse error after {max_retries} retries: {e}",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "Failed to parse model output.",
                "supporting_image_ids": "none",
                "valid_image": False,
                "severity": "unknown",
            }
        except Exception as e:
            err_msg = str(e)
            if "invalid_image_format" in err_msg or "invalid_request_error" in err_msg:
                return {
                    "evidence_standard_met": False,
                    "evidence_standard_met_reason": str(e),
                    "risk_flags": "manual_review_required",
                    "issue_type": "unknown",
                    "object_part": "unknown",
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "Image could not be processed by the model.",
                    "supporting_image_ids": "none",
                    "valid_image": False,
                    "severity": "unknown",
                }
            if attempt < max_retries - 1:
                delay = 5 * (2 ** attempt)
                time.sleep(delay)
                continue
            return {
                "evidence_standard_met": False,
                "evidence_standard_met_reason": f"API error after {max_retries} retries: {e}",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "API error.",
                "supporting_image_ids": "none",
                "valid_image": False,
                "severity": "unknown",
            }
