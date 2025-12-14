# Project : Borrower Risk Segmentation via Sparse PCA
#### Lending Club Accepted Loans (2007‑2018)

## Data
- Source: [Lending Club Accepted Loans (2007‑2018)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Files: `accepted_2007_to_2018Q4.csv` (~2 GB)
- **Note:** Due to size, the raw data is not stored in this repo. Download it from the source link above and place it in a `data/` folder to run the notebook.

1. **Business Objective**

Identify latent risk profiles in Lending Club borrowers using unsupervised learning, providing actionable segments for underwriting, marketing, or product design.

2. **Dataset Overview**

Source: Lending Club “accepted loans” (2007‑Q4 2018)

Initial size: 2,260,701 rows × 151 columns

Final cleaned features: 134 numeric columns (after encoding, imputation, and scaling)

3. **Data Preparation Pipeline**

3.1. Handling Object‑Type Columns (38 columns)

Category	Columns	Transformation
Drop (18)	id, url, desc, emp_title, title, zip_code, verification_status_joint, sec_app_earliest_cr_line, hardship_type, hardship_reason, hardship_status, hardship_start_date, hardship_end_date, payment_plan_start_date, hardship_loan_status, debt_settlement_flag_date, settlement_status, settlement_date	Removed (unique IDs, free text, >50% nulls)
Binary (3)	pymnt_plan, hardship_flag, debt_settlement_flag	Mapped {'Y':1, 'N':0} or {'y':1, 'n':0}
Date (5)	issue_d, earliest_cr_line, last_pymnt_d, next_pymnt_d, last_credit_pull_d	Converted to datetime, extracted numeric features (year, month, time‑delta)
Categorical (2‑level) (4)	term, initial_list_status, application_type, disbursement_method	Label‑encoded to {0,1}
Categorical (multi‑level) (8)	grade, sub_grade, emp_length, home_ownership, verification_status, loan_status, purpose, addr_state	Ordinal mapping (where logical) or label‑encoding

3.2. Numeric Cleaning

Dropped numeric columns with >50% nulls (e.g., member_id, most hardship/settlement fields).

Filled remaining nulls (≤0.1% per column) with column median.

Feature scaling: Standardized all continuous features (StandardScaler); binary features left unscaled.

3.3. Final Pre‑PCA Dataset

Shape: 2,260,701 rows × 134 columns

Data types: float64 (continuous), int64 (binary/encoded)

Nulls: 0

Correlation: High multicollinearity expected (e.g., loan_amnt ↔ funded_amnt), addressed via sparse PCA.

4. **Dimensionality Reduction: Sparse PCA**

4.1. Why Sparse PCA?

Standard PCA yields dense components where every feature contributes, hindering interpretation.

Sparse PCA imposes an L₁ penalty (alpha), forcing many loadings to zero.

Result: each component is driven by a small subset of features, making business labeling straightforward.

4.2. Model Configuration

python
SparsePCA(
    n_components=10,
    alpha=10,          # sparsity regularization
    random_state=42,   # reproducibility
    max_iter=1000
)
Sample used for fitting: 200,000 rows (random sample, representative).

Explained variance (10 components): 40.3%
Lower than standard PCA due to sparsity constraint, but trade‑off accepted for interpretability.

4.3. Component Loadings & Sparsity

Component	Non‑zero loadings (out of 134)	Dominant features (top 3)
0	58	num_op_rev_tl, num_actv_rev_tl, num_rev_tl_bal_gt_0
1	70	num_tl_op_past_12m, acc_open_past_24mths, open_rv_12m
2	69	fico_range_high, fico_range_low, bc_util
3	63	total_il_high_credit_limit, total_bal_il, total_bal_ex_mort
4	72	out_prncp, out_prncp_inv, issue_year
5	74	loan_amnt, funded_amnt, funded_amnt_inv
6	14	sec_app_collections_12_mths_ex_med, sec_app_chargeoff_within_12_mths, sec_app_fico_range_low
7	51	delinq_2yrs, mths_since_last_delinq, mths_since_recent_revol_delinq
8	33	recoveries, collection_recovery_fee, loan_status
9	7	hardship_amount, hardship_payoff_balance_amount, orig_projected_additional_accrued_interest
Components 6 and 9 are notably sparse (<20 non‑zeros), highlighting focused risk signals.

5. **Component Interpretation & Labeling**

Component	Assigned Label	Business Meaning
0	Active Credit Lines	Count of active revolving accounts and balances >0.
1	Recent Account Openings	New accounts opened in the last 6‑24 months.
2	Credit Score vs. Utilization	Trade‑off between high FICO scores and high credit‑card/utilization rates.
3	Installment Loan Exposure	Total balances and limits on installment loans.
4	Loan Aging & Outstanding Principal	Loan vintage (issue year) and remaining unpaid principal.
5	Loan Size & Payment	Original loan amount, funded amount, and monthly installment.
6	Joint‑Application Risk	Features from the secondary applicant (co‑borrower) on joint applications.
7	Recent Delinquency History	Recency and frequency of delinquencies (last 24 months).
8	Recovery & Default Status	Post‑charge‑off recoveries and current loan‑status flags.
9	Hardship Program Involvement	Amounts and balances related to borrower hardship programs.

6. **Visualization**

6.1. Biplot (Components 0 & 1)

../data/biplot.png
Arrow direction and length indicate feature contribution to components. Gray points are borrowers (subsampled for clarity).

Key observations:

Component 0 (x‑axis) separates borrowers by number of active credit lines.

Component 1 (y‑axis) separates borrowers by recent account openings.

Features such as num_op_rev_tl and num_tl_op_past_12m dominate the respective components, confirming our labels.

6.2. Borrower Distribution in PCA Space

../data/scatter.png
*Borrowers colored by loan status (0=Good, 1=Risky, 2=Bad).*

Lower‑left quadrant: Borrowers with few active lines and few recent openings—likely low‑risk, established borrowers.

Upper‑right quadrant: Borrowers with many active lines and recent openings—potential credit‑seeking, higher‑risk segment.

7. **Business Insights & Recommendations**

Risk‑based segmentation
The 10 components provide a multi‑dimensional risk profile beyond a simple credit score. Lending Club could:

Create risk tiers combining component scores (e.g., high delinquency + high utilization = highest risk).

Adjust pricing or loan limits per tier.

Targeted interventions

Component 7 (delinquency history): Flag borrowers with high scores for early‑stage collection outreach.

Component 9 (hardship program): Identify borrowers likely to need hardship options early.

Product design

Component 6 (joint‑application risk): Develop specific underwriting rules for co‑signed loans.

Component 2 (credit‑score‑vs‑utilization): Offer credit‑line increase campaigns to high‑score, low‑utilization borrowers.

Model deployment

Sparse PCA components can be calculated in real‑time (linear projection) for new applicants.

Each component is interpretable, supporting regulatory compliance (no “black box”).

8. **Limitations & Next Steps**

Limitations
Explained variance: 40.3% is moderate; additional components could be added but would reduce sparsity.

Sample bias: Fitted on 200k‑row sample; transformation applied to full dataset assumes representativeness.

Linear assumption: PCA captures linear relationships; nonlinear patterns may remain in residuals.

Next Steps
Clustering: Apply K‑means to the 10 component scores to define discrete borrower groups.

Supervised validation: Regress component scores against actual default rates to validate risk ordering.

Time‑series analysis: Track how component scores evolve for borrowers over multiple loans.

Production pipeline: Automate the cleaning and projection steps for new loan applications.

9. **Technical Environment**

Language: Python 3.8

Libraries: pandas, numpy, scikit‑learn, matplotlib, seaborn

Notebook: Jupyter in VS Code

Repository: https://github.com/madianis/data-science-portfolio

10. **Conclusion**

This project demonstrated a complete unsupervised learning workflow on a large, real‑world financial dataset. By applying Sparse PCA, we transformed 134 noisy features into 10 interpretable risk dimensions, each tied to clear business logic. The resulting segments provide a data‑driven foundation for risk assessment, customer targeting, and product strategy at Lending Club.
