# Insurance Fraud Detection with AI

First AI business-case exposure: detecting high-risk fraudulent vehicle insurance claims using **synthetic data** (5,000 claims).  

Fraud is relatively rare (~10%) but very costly. The goal was to build models that rank claims by fraud risk so investigators can focus on the riskiest subset instead of reviewing all claims.

## Repository Contents
- `AI_Insurance_Fraud.ipynb` — Python notebook with full data generation, model training, evaluation, and plots.
- `report.pdf` — Business-style report (LaTeX/Overleaf) that summarizes the project motivation, data, methodology, results, and take-home message.
- `claims_dataset.csv` - Synthetic dataset of claims in a simulated mid-sized Dutch motor insurance company.

## Data
Synthetic dataset representing a Dutch mid-sized motor insurance company:
- Features: claim amount, days to file, channel, policy tenure, prior claims, region, vehicle age, claim type.
- Target: `is_fraud` (~10% of claims labeled as fraud).

## Methods
- **Language/stack:** Python, pandas, numpy, matplotlib, seaborn, scikit-learn
- **Models tested:** Logistic Regression (LR), Random Forest (RF)
- **Process:** 80/20 train-test split, scaling numeric features, one-hot encoding categoricals
- **Metrics:** ROC AUC, Precision, Recall, F1
- **Business framing:** thresholds chosen by “flag rate” (e.g., investigators review top 25% highest-risk claims)

## Results (Test set, 1,000 claims @ 25% flag rate)
- **Logistic Regression** — AUC 0.898, Precision 0.332, Recall 0.798, F1 0.469  
- **Random Forest** — AUC 0.867, Precision 0.308, Recall 0.750, F1 0.437  

At 25% flag rate:
- Workload reduced by **75%** (250 vs 1,000 claims reviewed)  
- Precision improved from **10% → 33%** (3.3× higher hit rate)  
- Recall remained high (**~80% of fraud caught**)  

## Take-Home Message
Even a simple, interpretable model like Logistic Regression, when aligned with operational capacity, can dramatically reduce workload while maintaining high fraud detection — showing how AI can deliver immediate, tangible business value.
