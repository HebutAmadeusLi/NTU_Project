
Du Jiayi
G2506446E
Jiayi026@e.ntu.edu.sg

# FairCredit Jurisdiction Engine

A jurisdiction-aware RegTech prototype for credit scoring, fair lending, and algorithmic fairness across the United States and the European Union.

This project was developed for **MH6822 Regulatory Technology — Assignment 1**. It demonstrates how the same synthetic credit-scoring model can produce different compliance outputs depending on the selected jurisdiction.

## 1. Project Overview

**FairCredit Jurisdiction Engine** is a Python-based compliance-support prototype designed for a hypothetical RegTech provider serving the Chief Compliance Officer (CCO) and model risk teams of a global bank.

The selected regulated entity is **JPMorgan Chase & Co.** The selected domain is **credit scoring, fair lending, and algorithmic fairness**. The selected jurisdictions are the **United States** and the **European Union**.

The tool uses synthetic loan-application data, trains a simple logistic regression model, applies jurisdiction-specific compliance rules, and generates structured JSON reports.

## 2. Key Idea

A global bank cannot apply the same compliance checklist to every jurisdiction.

In the **United States**, the prototype focuses on:

- ECOA / Regulation B-style adverse-action explainability
- Fair-lending monitoring
- Consumer-facing adverse-action reason codes
- Internal disparate-impact review triggers

In the **European Union**, the prototype treats creditworthiness assessment as a high-risk AI use case and adds:

- Bias testing
- Technical documentation
- Audit logging
- Human oversight records
- High-risk AI governance evidence

This means the same model and the same dataset can lead to different report structures, thresholds, and compliance interpretations.

## 3. Repository Contents

```text
.
├── Task3_jurisdiction_aware_regtech_tool.py   # Main executable Python prototype
├── README.md                                  # Project documentation
├── task3_outputs/                             # Generated after running the script
│   ├── us_compliance_report.json
│   ├── eu_compliance_report.json
│   └── model_card.json
└── MH6822_Assignment1_Jurisdiction_Aware_RegTech_Revised.docx
```

The `task3_outputs/` folder is generated automatically when the Python script is run.

## 4. Requirements

The prototype requires Python 3.9 or above and the following Python packages:

```bash
pip install pandas numpy scikit-learn
```

## 5. How to Run

From the project folder, run:

```bash
python Task3_jurisdiction_aware_regtech_tool.py
```

The script will:

1. Generate a synthetic credit-application dataset.
2. Train a logistic regression credit approval model.
3. Build a United States compliance report.
4. Build a European Union compliance report.
5. Generate a model card / governance documentation stub.
6. Save all outputs into the `task3_outputs/` folder.

## 6. Expected Console Output

A successful run should produce output similar to:

```text
Generating synthetic credit data...
Training credit model...
Building US compliance report...
Building EU compliance report...
Building model card...

=== TASK 3 PROTOTYPE RUN COMPLETE ===
Model performance: {'test_accuracy': 0.6861, 'test_auc': 0.6553, 'test_size': 360, 'train_size': 840}
US status: REVIEW_REQUIRED
EU status: REVIEW_REQUIRED
Files saved in: task3_outputs

US report focuses on adverse-action reasons and fair-lending monitoring.
EU report adds high-risk AI governance, technical documentation, audit log and human oversight fields.
```

## 7. Generated Output Files

After running the script, three JSON files are generated.

### `us_compliance_report.json`

This report represents the United States compliance view. It focuses on:

- Regulation B / ECOA-style adverse-action explainability
- Consumer-facing reason codes for denied applicants
- Internal fair-lending monitoring
- Disparate-impact alert triggers

### `eu_compliance_report.json`

This report represents the European Union compliance view. It includes:

- High-risk AI classification for creditworthiness assessment
- Technical documentation requirements
- Human oversight record
- Audit-log requirement
- Bias testing and fairness monitoring

### `model_card.json`

This file provides a governance documentation stub for the model. It includes:

- Model name and model type
- Intended use and non-intended uses
- Input features
- Protected attributes used for audit only
- Model performance
- Assumptions
- Known failure modes

## 8. Core Model Design

The prototype trains a logistic regression classification model using synthetic loan-application data.

The model uses the following input features:

```text
income
credit_score
debt_to_income
delinquencies
requested_loan_amount
```

The following protected attributes are **not used for model training** but are retained for fairness auditing:

```text
gender
ethnicity
```

This design demonstrates an important compliance point: even when protected attributes are excluded from model training, model outcomes can still differ across groups because other variables may act as proxies or reflect structural differences in the synthetic data.

## 9. Jurisdiction Logic

The rules engine uses different compliance settings for the United States and the European Union.

| Jurisdiction | Main focus | Disparate-impact threshold | Main output style |
| --- | --- | ---: | --- |
| United States | Adverse-action explainability and fair-lending monitoring | 0.80 | Consumer-facing reason codes + internal fairness alert |
| European Union | High-risk AI governance for creditworthiness assessment | 0.90 | Technical documentation + bias testing + audit log + human oversight |

The thresholds are internal review triggers used for prototype demonstration. They are not legal safe harbours.

## 10. Actual Prototype Result

In the submitted run, both jurisdiction reports returned:

```text
REVIEW_REQUIRED
```

This happened because the ethnicity-based disparate-impact ratio was below the relevant review threshold in both jurisdictions.

The result does not mean the model is legally unlawful. It means the tool has identified a fairness risk that should be reviewed by a human compliance or model-risk team.

The key difference is how each jurisdiction interprets the alert:

- In the **United States**, it is treated mainly as a fair-lending and adverse-action explanation review issue.
- In the **European Union**, it is treated as part of a broader high-risk AI governance review requiring documentation, audit logging, and human oversight.

## 11. Validating the JSON Outputs

To check that the generated JSON files are valid, run:

```bash
python -m json.tool task3_outputs/us_compliance_report.json
python -m json.tool task3_outputs/eu_compliance_report.json
python -m json.tool task3_outputs/model_card.json
```

If each command prints formatted JSON without an error, the files are valid.

## 12. Limitations

This prototype is designed for classroom demonstration only. It has several important limitations:

- It uses synthetic data, not real bank credit data.
- It does not make real credit decisions.
- It does not provide legal advice.
- The adverse-action explanations are coefficient-based approximations, not causal explanations.
- The EU AI Act lifecycle is simplified into a small governance stub.
- The fairness thresholds are internal review triggers, not legal conclusions.
- The prototype does not automatically update rules when regulations change.

## 13. Possible Future Improvements

Future versions could add:

- Real anonymised credit-performance data
- SHAP or LIME explanations
- Confidence intervals for fairness metrics
- More subgroup and intersectional fairness testing
- Model drift monitoring over time
- A user interface for compliance review and sign-off
- Versioned regulatory rule updates
- Stronger audit-trail and evidence-management features

## 14. Academic Use Notice

This project is a prototype for academic coursework. It is intended to demonstrate jurisdiction-aware RegTech design logic, not to serve as a production credit-scoring system or a legal compliance opinion.
