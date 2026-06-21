import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from claim_processor import process_claim, load_evidence_reqs_text

load_dotenv()

_REPO_ROOT = os.path.dirname(_CODE_DIR)
DATASET_DIR = os.path.join(_REPO_ROOT, "dataset")
CLAIMS_CSV = os.path.join(DATASET_DIR, "claims.csv")
USER_HISTORY_CSV = os.path.join(DATASET_DIR, "user_history.csv")
EVIDENCE_REQS_CSV = os.path.join(DATASET_DIR, "evidence_requirements.csv")
OUTPUT_CSV = os.path.join(_REPO_ROOT, "output.csv")

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def main():
    print(f"Loading claims from {CLAIMS_CSV}")
    claims_df = pd.read_csv(CLAIMS_CSV)

    print(f"Loading user history from {USER_HISTORY_CSV}")
    user_history_df = pd.read_csv(USER_HISTORY_CSV)

    print(f"Loading evidence requirements from {EVIDENCE_REQS_CSV}")
    evidence_reqs_df = pd.read_csv(EVIDENCE_REQS_CSV)
    evidence_reqs_text = load_evidence_reqs_text(EVIDENCE_REQS_CSV)

    results = []
    total = len(claims_df)

    for idx, (_, claim_row) in enumerate(claims_df.iterrows()):
        row_dict = claim_row.to_dict()
        user_id = row_dict.get("user_id", "")
        print(f"[{idx + 1}/{total}] Processing claim for user {user_id}...")

        start = time.time()
        pred = process_claim(
            row=row_dict,
            user_history_df=user_history_df,
            evidence_reqs_df=evidence_reqs_df,
            evidence_reqs_text=evidence_reqs_text,
            dataset_dir=DATASET_DIR,
        )
        elapsed = time.time() - start

        out_row = {
            "user_id": user_id,
            "image_paths": row_dict.get("image_paths", ""),
            "user_claim": row_dict.get("user_claim", ""),
            "claim_object": row_dict.get("claim_object", ""),
            **pred,
        }
        results.append(out_row)
        print(f"  -> claim_status={out_row['claim_status']}, severity={out_row['severity']} ({elapsed:.1f}s)")

        time.sleep(1.5)

    out_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    out_df.to_csv(OUTPUT_CSV, index=False, quoting=1)
    print(f"\nDone. Wrote {len(out_df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
