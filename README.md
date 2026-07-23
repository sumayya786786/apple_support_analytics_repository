# Apple Retail Support Customer Feedback Analytics

Reproducible code and aggregate outputs for the dissertation **Using Customer Feedback Data to Identify the Key Drivers of Satisfaction and Dissatisfaction in Apple Retail Support Services**.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --input data/apple_support_survey_data.xlsx
```

Place the anonymised `Responses` worksheet at `data/apple_support_survey_data.xlsx`. The pipeline verifies the schema and SHA-256 checksum, never overwrites the input, and writes aggregate results to `outputs/`.

## Complete analytical coverage

The pipeline generates:

- consent, eligibility, range, missingness and duplicate audits;
- sample profile, satisfaction distribution, class balance and convergent correlations;
- service-attribute means, poor-rating shares and Spearman correlations;
- Q14/Q15 reliability evidence;
- resolution-group means, Kruskal-Wallis, epsilon-squared and Holm-adjusted pairwise tests;
- HC3-robust OLS, unstandardised effects, continuous standardised betas, incremental and partial R-squared;
- VIF, heteroscedasticity, normality, influence, Huber and case-removal sensitivity checks;
- turnaround-applicable modelling, exploratory extensions and elastic-net sensitivity;
- repeated nested stratified validation for regularised logistic regression and random forest;
- fixed-threshold and inner-F1 threshold results, confusion matrices, calibration, Brier scores and sensitivity target analysis;
- validation permutation importance and robust binomial odds ratios for H4b;
- reproducible hashed manual thematic coding, duplicate-wording sensitivity and respondent-selected factor counts;
- every analytical figure used in Chapter 4 plus supplementary diagnostics.

## Repository structure

- `run_all.py`: executes the full pipeline.
- `config/model_config.json`: genuinely controls seeds, folds, grids and model settings.
- `src/prepare.py`: schema enforcement, inclusion rules, quality audit and derived variables.
- `src/eda.py`: descriptive, reliability, contextual and non-parametric analysis.
- `src/explanatory.py`: OLS, effect importance, diagnostics and sensitivity models.
- `src/classification.py`: nested validation, calibration, sensitivity and H4b analysis.
- `src/text_coding.py`: hashed single-researcher multi-label coding and sensitivity counts.
- `appendix_materials/`: data dictionary, codebook, decisions and coding hashes.
- `outputs/tables/output_index.csv`: inventory of generated outputs.
- `outputs/tables/headline_results.json`: machine-readable headline results.
- `tests/test_pipeline.py`: core integrity and output-coverage checks.
