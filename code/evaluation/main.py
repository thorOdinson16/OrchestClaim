import os
import sys
import time

import pandas as pd

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from claim_processor import process_claim, load_evidence_reqs_text

_REPO_ROOT = os.path.dirname(_CODE_DIR)
DATASET_DIR = os.path.join(_REPO_ROOT, "dataset")
SAMPLE_CSV = os.path.join(DATASET_DIR, "sample_claims.csv")
USER_HISTORY_CSV = os.path.join(DATASET_DIR, "user_history.csv")
EVIDENCE_REQS_CSV = os.path.join(DATASET_DIR, "evidence_requirements.csv")

COMPARE_FIELDS = [
    ("evidence_standard_met", "Evidence Standard Met"),
    ("issue_type", "Issue Type"),
    ("object_part", "Object Part"),
    ("claim_status", "Claim Status"),
    ("valid_image", "Valid Image"),
    ("severity", "Severity"),
]


def evaluate():
    print(f"Loading sample claims from {SAMPLE_CSV}")
    sample_df = pd.read_csv(SAMPLE_CSV)

    print(f"Loading user history from {USER_HISTORY_CSV}")
    user_history_df = pd.read_csv(USER_HISTORY_CSV)

    print(f"Loading evidence requirements from {EVIDENCE_REQS_CSV}")
    evidence_reqs_df = pd.read_csv(EVIDENCE_REQS_CSV)
    evidence_reqs_text = load_evidence_reqs_text(EVIDENCE_REQS_CSV)

    total = len(sample_df)
    correct = {field: 0 for field, _ in COMPARE_FIELDS}
    total_errors = 0

    claim_status_confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "details": []}

    for idx, (_, sample_row) in enumerate(sample_df.iterrows()):
        row_dict = sample_row.to_dict()
        user_id = row_dict.get("user_id", "")
        print(f"[{idx + 1}/{total}] Evaluating claim for user {user_id}...")

        start = time.time()
        pred = process_claim(
            row=row_dict,
            user_history_df=user_history_df,
            evidence_reqs_df=evidence_reqs_df,
            evidence_reqs_text=evidence_reqs_text,
            dataset_dir=DATASET_DIR,
        )
        elapsed = time.time() - start

        mismatches = []
        for field, label in COMPARE_FIELDS:
            expected = str(row_dict.get(field, "")).strip().lower()
            predicted = str(pred.get(field, "")).strip().lower()

            if field == "risk_flags":
                exp_set = set(f.strip() for f in expected.split(";") if f.strip())
                pred_set = set(f.strip() for f in predicted.split(";") if f.strip())
                if exp_set == pred_set:
                    correct[field] += 1
                else:
                    mismatches.append(f"  {label}: expected={expected}, got={predicted}")
            elif field == "evidence_standard_met":
                if expected == predicted:
                    correct[field] += 1
                else:
                    mismatches.append(f"  {label}: expected={expected}, got={predicted}")
            elif field == "valid_image":
                if expected == predicted:
                    correct[field] += 1
                else:
                    mismatches.append(f"  {label}: expected={expected}, got={predicted}")
            else:
                if expected == predicted:
                    correct[field] += 1
                else:
                    mismatches.append(f"  {label}: expected={expected}, got={predicted}")

        if mismatches:
            total_errors += 1
            print(f"  Mismatches for user {user_id} (expected -> predicted):")
            for m in mismatches:
                print(m)
            print()

        expected_status = str(row_dict.get("claim_status", "")).strip().lower()
        predicted_status = str(pred.get("claim_status", "")).strip().lower()
        claim_status_confusion["details"].append(
            {
                "user_id": user_id,
                "expected": expected_status,
                "predicted": predicted_status,
            }
        )

        time.sleep(1.5)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for field, label in COMPARE_FIELDS:
        accuracy = correct[field] / total * 100
        print(f"  {label:40s}: {correct[field]:3d}/{total:3d} = {accuracy:5.1f}%")

    overall_correct = sum(correct.values())
    overall_total = total * len(COMPARE_FIELDS)
    overall_accuracy = overall_correct / overall_total * 100
    print(f"  {'Overall':40s}: {overall_correct:3d}/{overall_total:3d} = {overall_accuracy:5.1f}%")

    print("\n" + "-" * 60)
    print("CLAIM STATUS CONFUSION MATRIX")
    statuses = ["supported", "contradicted", "not_enough_information"]
    cm = {e: {p: 0 for p in statuses} for e in statuses}
    for d in claim_status_confusion["details"]:
        exp = d["expected"] if d["expected"] in statuses else "unknown"
        pred = d["predicted"] if d["predicted"] in statuses else "unknown"
        if exp in cm and pred in cm[exp]:
            cm[exp][pred] += 1

    print(f"{'':>25} {'Predicted':>40}")
    print(f"{'Expected':<20}", end="")
    for p in statuses:
        print(f"{p:>18}", end="")
    print()
    for e in statuses:
        print(f"{e:<20}", end="")
        for p in statuses:
            print(f"{cm[e][p]:>18}", end="")
        print()

    print(f"\nTotal claims evaluated: {total}")
    print(f"Claims with >=1 mismatch: {total_errors}")

    overall_claim_status_accuracy = (
        sum(cm[e][e] for e in statuses) / total * 100
    )
    print(f"Claim status accuracy: {sum(cm[e][e] for e in statuses)}/{total} = {overall_claim_status_accuracy:.1f}%")

    return overall_accuracy


if __name__ == "__main__":
    evaluate()
