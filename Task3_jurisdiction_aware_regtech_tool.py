"""
Task3_jurisdiction_aware_regtech_tool.py

MH6822 Regulatory Technology — Task 3 Working Prototype
Jurisdiction-aware RegTech tool for credit-scoring / algorithmic fairness.

Chosen entity example: JPMorgan Chase
Domain: Fair lending / algorithmic fairness / credit scoring
Jurisdictions: United States vs European Union

Run:
    pip install pandas numpy scikit-learn
    python Task3_jurisdiction_aware_regtech_tool.py

Outputs:
    task3_outputs/us_compliance_report.json
    task3_outputs/eu_compliance_report.json
    task3_outputs/model_card.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. Jurisdiction configuration layer
JURISDICTION_RULES: Dict[str, Dict] = {
    "US": {
        "name": "United States",
        "regulatory_focus": "Regulation B / ECOA adverse-action explainability and fair lending monitoring",
        "high_risk_ai_classification": False,
        "approval_threshold": 0.50,
        "fairness_metric": "disparate_impact_ratio",
        "disparate_impact_min_ratio": 0.80,
        "requires_adverse_action_reasons": True,
        "requires_technical_documentation": False,
        "requires_human_oversight_record": False,
        "requires_audit_log": True,
        "reporting_style": "consumer-facing adverse action summary + internal fairness alert",
    },
    "EU": {
        "name": "European Union",
        "regulatory_focus": "EU AI Act high-risk AI system governance for creditworthiness assessment",
        "high_risk_ai_classification": True,
        "approval_threshold": 0.50,
        "fairness_metric": "disparate_impact_ratio",
        "disparate_impact_min_ratio": 0.90,
        "requires_adverse_action_reasons": True,
        "requires_technical_documentation": True,
        "requires_human_oversight_record": True,
        "requires_audit_log": True,
        "reporting_style": "technical documentation + bias testing + audit log + human oversight record",
    },
}


def generate_synthetic_credit_data(n: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic loan application dataset.

    Protected attributes are generated for fairness auditing, but they are NOT used
    as model input features. This imitates a compliance situation where a model may
    not directly use protected attributes, but outcomes can still differ across groups.
    """
    rng = np.random.default_rng(random_state)

    gender = rng.choice(["female", "male"], size=n, p=[0.52, 0.48])
    ethnicity = rng.choice(["majority_group", "minority_group"], size=n, p=[0.72, 0.28])

    # Synthetic structural differences for a fairness-monitoring demonstration only.
    base_income = rng.normal(75000, 18000, size=n)
    income_adjustment = np.where(ethnicity == "minority_group", -6000, 0)
    income = np.clip(base_income + income_adjustment, 25000, 180000)

    base_credit_score = rng.normal(680, 55, size=n)
    score_adjustment = np.where(ethnicity == "minority_group", -18, 0)
    credit_score = np.clip(base_credit_score + score_adjustment, 300, 850)

    debt_to_income = np.clip(rng.normal(0.35, 0.12, size=n), 0.05, 0.85)
    delinquencies = rng.poisson(0.35, size=n)
    requested_loan_amount = np.clip(rng.normal(180000, 60000, size=n), 30000, 500000)

    # Ground-truth synthetic approval probability using financial variables only.
    logit = (
        0.0
        + 0.0065 * (credit_score - 600)
        + 0.000018 * (income - 50000)
        - 2.4 * debt_to_income
        - 0.42 * delinquencies
        - 0.000003 * requested_loan_amount
    )
    approval_probability = 1 / (1 + np.exp(-logit))
    approved = rng.binomial(1, approval_probability)

    return pd.DataFrame(
        {
            "applicant_id": [f"A{i:05d}" for i in range(n)],
            "income": income.round(2),
            "credit_score": credit_score.round(0).astype(int),
            "debt_to_income": debt_to_income.round(3),
            "delinquencies": delinquencies.astype(int),
            "requested_loan_amount": requested_loan_amount.round(2),
            "gender": gender,
            "ethnicity": ethnicity,
            "approved": approved.astype(int),
        }
    )


MODEL_FEATURES = [
    "income",
    "credit_score",
    "debt_to_income",
    "delinquencies",
    "requested_loan_amount",
]
PROTECTED_ATTRIBUTES = ["gender", "ethnicity"]


def train_credit_model(df: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, Dict]:
    """Train a simple logistic regression model."""
    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=7,
        stratify=df["approved"],
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(train_df[MODEL_FEATURES], train_df["approved"])

    test_proba = model.predict_proba(test_df[MODEL_FEATURES])[:, 1]
    test_pred = (test_proba >= 0.50).astype(int)

    performance = {
        "test_accuracy": round(float(accuracy_score(test_df["approved"], test_pred)), 4),
        "test_auc": round(float(roc_auc_score(test_df["approved"], test_proba)), 4),
        "test_size": int(len(test_df)),
        "train_size": int(len(train_df)),
    }
    return model, train_df, test_df.copy(), performance


def get_predictions(model: Pipeline, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add predicted probability and predicted approval columns."""
    out = df.copy()
    out["predicted_approval_probability"] = model.predict_proba(out[MODEL_FEATURES])[:, 1]
    out["predicted_approved"] = (out["predicted_approval_probability"] >= threshold).astype(int)
    return out


def approval_rate_by_group(df: pd.DataFrame, group_col: str) -> Dict[str, float]:
    rates = df.groupby(group_col)["predicted_approved"].mean().to_dict()
    return {k: round(float(v), 4) for k, v in rates.items()}


def disparate_impact_ratio(df: pd.DataFrame, group_col: str) -> float:
    """lowest group approval rate / highest group approval rate."""
    rates = df.groupby(group_col)["predicted_approved"].mean()
    if len(rates) < 2 or rates.max() == 0:
        return 0.0
    return round(float(rates.min() / rates.max()), 4)


def demographic_parity_difference(df: pd.DataFrame, group_col: str) -> float:
    rates = df.groupby(group_col)["predicted_approved"].mean()
    if len(rates) < 2:
        return 0.0
    return round(float(rates.max() - rates.min()), 4)


def false_negative_rate_by_group(df: pd.DataFrame, group_col: str) -> Dict[str, float]:
    result = {}
    for group, sub in df.groupby(group_col):
        positives = sub[sub["approved"] == 1]
        if len(positives) == 0:
            result[group] = None
        else:
            fnr = ((positives["predicted_approved"] == 0).sum()) / len(positives)
            result[group] = round(float(fnr), 4)
    return result


def coefficient_explanation(model: Pipeline, applicant_row: pd.Series, top_n: int = 3) -> List[Dict]:
    """
    Produce simple reason codes from logistic regression coefficients.
    This is a lightweight prototype explanation, not a causal explanation.
    """
    scaler: StandardScaler = model.named_steps["scaler"]
    classifier: LogisticRegression = model.named_steps["classifier"]

    x = applicant_row[MODEL_FEATURES].to_frame().T
    x_scaled = scaler.transform(x)[0]
    coefficients = classifier.coef_[0]
    contributions = x_scaled * coefficients

    rows = []
    for feature, contribution, raw_value in zip(MODEL_FEATURES, contributions, applicant_row[MODEL_FEATURES]):
        rows.append(
            {
                "feature": feature,
                "raw_value": float(raw_value),
                "contribution_to_approval": round(float(contribution), 4),
            }
        )
    return sorted(rows, key=lambda r: r["contribution_to_approval"])[:top_n]


def fairness_summary(df: pd.DataFrame) -> Dict:
    summary = {}
    for attr in PROTECTED_ATTRIBUTES:
        summary[attr] = {
            "approval_rate_by_group": approval_rate_by_group(df, attr),
            "disparate_impact_ratio": disparate_impact_ratio(df, attr),
            "demographic_parity_difference": demographic_parity_difference(df, attr),
            "false_negative_rate_by_group": false_negative_rate_by_group(df, attr),
        }
    return summary


def build_compliance_report(
    jurisdiction: str,
    model: Pipeline,
    test_df: pd.DataFrame,
    model_performance: Dict,
) -> Dict:
    """Build a jurisdiction-specific compliance report."""
    if jurisdiction not in JURISDICTION_RULES:
        raise ValueError(f"Unsupported jurisdiction: {jurisdiction}. Use 'US' or 'EU'.")

    rules = JURISDICTION_RULES[jurisdiction]
    scored_df = get_predictions(model, test_df, threshold=rules["approval_threshold"])
    fairness = fairness_summary(scored_df)

    fairness_alerts = []
    for attr, metrics in fairness.items():
        ratio = metrics["disparate_impact_ratio"]
        if ratio < rules["disparate_impact_min_ratio"]:
            fairness_alerts.append(
                {
                    "protected_attribute": attr,
                    "metric": "disparate_impact_ratio",
                    "observed_value": ratio,
                    "threshold": rules["disparate_impact_min_ratio"],
                    "alert": "Potential disparity detected; compliance team review required.",
                }
            )

    denied = scored_df[scored_df["predicted_approved"] == 0].copy()
    if len(denied) > 0:
        sample_denial = denied.sort_values("predicted_approval_probability").iloc[0]
        adverse_action_reasons = coefficient_explanation(model, sample_denial, top_n=3)
        sample_denied_applicant = {
            "applicant_id": sample_denial["applicant_id"],
            "predicted_approval_probability": round(float(sample_denial["predicted_approval_probability"]), 4),
            "top_adverse_reasons": adverse_action_reasons,
        }
    else:
        sample_denied_applicant = None

    report = {
        "jurisdiction": jurisdiction,
        "jurisdiction_name": rules["name"],
        "regulatory_focus": rules["regulatory_focus"],
        "model_performance": model_performance,
        "fairness_summary": fairness,
        "fairness_alerts": fairness_alerts,
        "sample_denied_applicant_explanation": sample_denied_applicant,
        "decision_boundary": {
            "approval_threshold": rules["approval_threshold"],
            "meaning": "Applicants with predicted approval probability >= threshold are predicted approved.",
        },
    }

    if jurisdiction == "US":
        report["us_specific_output"] = {
            "adverse_action_notice_required": rules["requires_adverse_action_reasons"],
            "consumer_facing_reason_codes": (
                sample_denied_applicant["top_adverse_reasons"] if sample_denied_applicant else []
            ),
            "compliance_interpretation": (
                "The US report focuses on whether the credit model can produce specific, "
                "accurate reasons for adverse credit decisions and whether internal fairness "
                "metrics flag potential disparate outcomes."
            ),
        }

    elif jurisdiction == "EU":
        report["eu_specific_output"] = {
            "high_risk_ai_system": rules["high_risk_ai_classification"],
            "technical_documentation_required": rules["requires_technical_documentation"],
            "human_oversight_required": rules["requires_human_oversight_record"],
            "audit_log_required": rules["requires_audit_log"],
            "human_oversight_record": {
                "review_owner": "Senior Credit Risk Manager",
                "override_allowed": True,
                "mandatory_review_trigger": "Any fairness alert or material model drift event",
            },
            "technical_documentation_stub": {
                "intended_purpose": "Credit approval support for consumer lending applications",
                "input_data": MODEL_FEATURES,
                "excluded_from_model_but_used_for_audit": PROTECTED_ATTRIBUTES,
                "known_limitations": [
                    "Synthetic training data may not match live applicant population.",
                    "Coefficient explanation is an approximation and not a full causal explanation.",
                    "The tool covers fairness monitoring, not the whole EU AI Act compliance lifecycle.",
                ],
                "data_governance_checks": [
                    "Check missingness by protected attribute.",
                    "Monitor approval-rate disparity by protected attribute.",
                    "Monitor performance drift and false negative rate differences.",
                ],
            },
            "compliance_interpretation": (
                "The EU report adds high-risk AI governance documentation, bias testing, "
                "audit logging and human oversight fields rather than only providing a consumer-facing denial reason."
            ),
        }

    report["prototype_compliance_status"] = (
        "REVIEW_REQUIRED" if fairness_alerts else "NO_FAIRNESS_ALERTS_IN_SYNTHETIC_TEST"
    )
    return report


def build_model_card(model_performance: Dict) -> Dict:
    return {
        "model_name": "Synthetic Credit Approval Logistic Regression Model",
        "model_type": "Logistic regression classification model",
        "intended_use": (
            "Classroom prototype for jurisdiction-aware RegTech compliance monitoring "
            "in credit scoring / algorithmic fairness."
        ),
        "not_intended_for": [
            "Real credit decisions",
            "Production lending",
            "Legal advice",
            "Full replacement of human compliance judgement",
        ],
        "input_features": MODEL_FEATURES,
        "protected_attributes_for_audit_only": PROTECTED_ATTRIBUTES,
        "performance": model_performance,
        "assumptions": [
            "Synthetic data is used because real bank credit data is unavailable.",
            "Protected attributes are excluded from model training but retained for fairness auditing.",
            "Different jurisdictions can require different reporting and governance outputs.",
        ],
        "failure_modes": [
            {
                "failure": "Jurisdictional misconfiguration",
                "example": "EU portfolio is checked using US-only report settings.",
                "mitigation": "Use validated jurisdiction codes and require compliance-owner approval.",
            },
            {
                "failure": "Model drift",
                "example": "Applicant population changes and approval disparity increases.",
                "mitigation": "Schedule periodic drift monitoring and retraining review.",
            },
            {
                "failure": "Rule change mid-period",
                "example": "Regulatory threshold or documentation requirement changes during a reporting cycle.",
                "mitigation": "Version the rules engine and store the rule version used for each report.",
            },
            {
                "failure": "Explanation overclaim",
                "example": "Reason codes are treated as causal explanations.",
                "mitigation": "Label explanations as model-based approximations and require human review.",
            },
        ],
    }


def main() -> None:
    output_dir = Path("task3_outputs")
    output_dir.mkdir(exist_ok=True)

    print("Generating synthetic credit data...")
    df = generate_synthetic_credit_data(n=1200, random_state=42)

    print("Training credit model...")
    model, train_df, test_df, performance = train_credit_model(df)

    print("Building US compliance report...")
    us_report = build_compliance_report("US", model, test_df, performance)

    print("Building EU compliance report...")
    eu_report = build_compliance_report("EU", model, test_df, performance)

    print("Building model card...")
    model_card = build_model_card(performance)

    (output_dir / "us_compliance_report.json").write_text(json.dumps(us_report, indent=2), encoding="utf-8")
    (output_dir / "eu_compliance_report.json").write_text(json.dumps(eu_report, indent=2), encoding="utf-8")
    (output_dir / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")

    print("\n=== TASK 3 PROTOTYPE RUN COMPLETE ===")
    print(f"Model performance: {performance}")
    print(f"US status: {us_report['prototype_compliance_status']}")
    print(f"EU status: {eu_report['prototype_compliance_status']}")
    print(f"Files saved in: {output_dir.resolve()}")
    print("\nUS report focuses on adverse-action reasons and fair-lending monitoring.")
    print("EU report adds high-risk AI governance, technical documentation, audit log and human oversight fields.")


if __name__ == "__main__":
    main()
