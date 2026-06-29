# Employee Attrition & Sentiment Analytics

A two-part HR analytics project combining a **structured machine learning model** (predicting individual employee attrition risk) with an **unstructured NLP model** (analyzing sentiment in public Glassdoor reviews). The two models are independent and answer different questions — see [Project Structure](#project-structure) for why they aren't combined into a single score.

---

## Overview

| | Part 1: Attrition Prediction | Part 2: Sentiment Analysis |
|---|---|---|
| **Question answered** | Is this specific employee likely to leave? | What do employees broadly say, and how positive/negative is it? |
| **Data** | IBM HR Analytics Employee Attrition (1,470 employees) | Glassdoor reviews (50,000 sampled from 838,566), scoped to Tech, Consulting, and Finance firms |
| **Technique** | Logistic Regression (with Random Forest / XGBoost compared via cross-validation) | TF-IDF + Logistic Regression |
| **Output** | Attrition probability + risk tier for a single employee | Sentiment label + confidence for any pasted review, plus aggregate firm-level sentiment |

---

## Project Structure

```
.
├── app.py                              # Streamlit app (both modules)
├── attrition_modelling.ipynb           # Part 1: ML pipeline
├── EDA.ipynb                           # Part 1: exploratory data analysis
├── glassdoor_sentiment_nlp.ipynb       # Part 2: NLP pipeline
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── glassdoor_reviews.csv               # not included (838K rows) — see Data section
│
├── attrition_model.pkl                 # trained Logistic Regression model
├── scaler.pkl                          # fitted StandardScaler
├── encoder.pkl                         # fitted OneHotEncoder
├── feature_names.pkl                   # ordered feature list expected by the model
├── threshold.pkl                       # tuned decision threshold
│
├── glassdoor_sentiment_pipeline.pkl    # TF-IDF + Logistic Regression pipeline
├── firm_sentiment_leaderboard.pkl      # per-firm sentiment, test-set only
└── glassdoor_nlp_config.json           # model metadata (AUC, scope, firm list)
```

### Why two separate models instead of one combined score

The IBM dataset and the Glassdoor dataset have no real-world link — there is no shared employee ID, and the IBM employees did not write the Glassdoor reviews in this dataset. Building a single "combined risk score" would require pairing rows from the two datasets arbitrarily, which would create a misleading correlation rather than a real one. Keeping the two models independent keeps every reported result honest:

- **Part 1** gives a precise, individual-level prediction, usable for one named employee.
- **Part 2** gives an aggregate, population-level signal — useful for understanding themes and sentiment trends across companies, not for predicting any specific person's behavior.

---

## Part 1: Attrition Prediction

### Data
IBM HR Analytics Employee Attrition dataset — 1,470 employee records, 35 original columns. Target: `Attrition` (Yes/No), ~84% Stay / ~16% Leave (imbalanced).

### Pipeline (`attrition_modelling.ipynb`)
1. Drop constant/identifier columns (`EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`) and misaligned compensation columns (`DailyRate`, `HourlyRate`, `MonthlyRate`).
2. Binary-encode `Attrition`, `OverTime`, `Gender`.
3. Engineer `Total_Satisfaction` as the mean of five correlated satisfaction sub-scores.
4. Stratified 70/30 train-test split (preserves the 84:16 class ratio).
5. One-hot encode categorical features — fit on training data only, to prevent leakage.
6. Check multicollinearity via VIF before trusting any Logistic Regression coefficient.
7. Scale features (Logistic Regression only — tree models are trained on unscaled data).
8. Compare Logistic Regression, Random Forest, and XGBoost via stratified 5-fold cross-validation.
9. Tune the classification threshold using the precision-recall curve (default 0.50 is not optimal for imbalanced data).
10. Evaluate on a held-out test set; inspect feature importance, ROC curve, and confusion matrix.

### Results
- **Final model:** Logistic Regression (selected over Random Forest/XGBoost and over a VIF-reduced variant, since it preserved better sensitivity to the minority "Leave" class)
- **Test AUC-ROC:** 0.815
- **Top features increasing attrition risk:** OverTime, frequent business travel, Sales roles, longer time since last promotion
- **Top features decreasing attrition risk:** higher Total Satisfaction, more Total Working Years

### EDA
See `EDA.ipynb` for the full exploratory analysis (target distribution, univariate/bivariate analysis, correlation heatmap, attrition rate by category). The summary at the end of that notebook cross-checks EDA findings against the model's feature importance.

---

## Part 2: Glassdoor Sentiment Analysis

### Data
Glassdoor employee reviews — 838,566 raw rows, sampled down to **50,000 rows** via stratified sampling (by firm and rating) after scoping to **Tech, Consulting, and Finance** firms only. Retail, food service, and hospitality firms were excluded, since they represent a structurally different employment population (hourly wages, shift work) than the salaried roles this project focuses on. The firm allow-list is documented in `glassdoor_nlp_config.json`.

3-star reviews were excluded as ambiguous; reviews rated 4-5 are labeled Positive, reviews rated 1-2 are labeled Negative.

### Pipeline (`glassdoor_sentiment_nlp.ipynb`)
1. Drop rows with missing `pros`/`cons`/`overall_rating`.
2. Label reviews Positive/Negative (3-star excluded), scope to allowed sectors, stratified-sample to 50,000 rows.
3. Clean text (lowercase, strip non-alphabetic characters, preserve apostrophes for contractions like "can't", "won't").
4. Combine `pros` and `cons` into a single `review_text` document per review, equally weighted.
5. Stratified 80/20 train-test split.
6. Vectorize with TF-IDF (`max_features=5000`, unigrams + bigrams, `min_df=3`, `max_df=0.90`, sublinear term-frequency scaling).
7. Train Logistic Regression (`class_weight='balanced'` to handle the 84:16 class imbalance) inside a single `sklearn.Pipeline`, so the vectorizer is always refit only on training data during cross-validation — preventing leakage.
8. Validate with stratified 5-fold cross-validation; evaluate once on the held-out test set.
9. Extract top positive/negative TF-IDF features from the model's learned coefficients.
10. Build a per-firm sentiment leaderboard, using **test-set predictions only** (so no firm's score is inflated by the model having already trained on those exact reviews); firms with fewer than 50 test reviews are excluded as too noisy to report reliably.

### Results
- **Model:** TF-IDF + Logistic Regression
- **Test AUC-ROC:** *(insert your final number from `glassdoor_nlp_config.json`)*
- **Cross-validation AUC:** *(insert mean ± std from `glassdoor_nlp_config.json`)*
- **Class balance handled via** `class_weight='balanced'`, not by discarding data — large corporate employers naturally generate far fewer negative reviews than positive ones (84:16 split), and this pattern held even after broadening scope from Tech-only to Tech+Consulting+Finance.

---

## The Streamlit App (`app.py`)

A single two-tab application built on the saved model artifacts from both parts.

**Tab 1 — Employee Attrition Predictor**
- EDA summary (target distribution, attrition drivers by category, key numeric differences, correlation heatmap) rendered live from the raw IBM CSV.
- A full input form for an individual employee's HR attributes.
- On submit: attrition probability, a High/Moderate/Low risk badge (thresholds tuned from the model), and a list of contributing risk factors with suggested retention actions.

**Tab 2 — Glassdoor Sentiment Analyzer**
- About the data: scope, review count, companies covered, class split.
- What we did: plain-language explanation of the TF-IDF + Logistic Regression approach.
- Results: AUC metrics, a bar chart of the top words driving positive/negative predictions, and the per-firm sentiment leaderboard.
- Try it live: paste any review text and get an instant sentiment prediction, confidence score, and highlighted positive/negative trigger words.

### Running the app

The trained model files (`.pkl`) are **not included in this repository** — they must be generated locally by running the two notebooks first.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate model artifacts (run all cells in each notebook, in order)
#    - attrition_modelling.ipynb  -> attrition_model.pkl, scaler.pkl, encoder.pkl,
#                                     feature_names.pkl, threshold.pkl
#    - glassdoor_sentiment_nlp.ipynb -> glassdoor_sentiment_pipeline.pkl,
#                                        firm_sentiment_leaderboard.pkl,
#                                        glassdoor_nlp_config.json

# 3. Run the app
streamlit run app.py
```

Ensure the following files end up in the same folder as `app.py` before running it:
`attrition_model.pkl`, `scaler.pkl`, `encoder.pkl`, `feature_names.pkl`, `threshold.pkl`, `glassdoor_sentiment_pipeline.pkl`, `firm_sentiment_leaderboard.pkl`, `glassdoor_nlp_config.json`, `WA_Fn-UseC_-HR-Employee-Attrition.csv`.

### Data Source

- `WA_Fn-UseC_-HR-Employee-Attrition.csv` is included in this repo.
- `glassdoor_reviews.csv` (838,566 rows) is **not included** due to size and dataset licensing terms. *(Insert the source link you used — e.g. Kaggle — here.)* Place the downloaded file in the project root before running `glassdoor_sentiment_nlp.ipynb`.

---

## Key Design Decisions & Honest Limitations

- **No combined risk score.** As explained above, the two datasets cannot be linked at the row level, so they remain two parallel, independently-validated models rather than one fused score.
- **3-star Glassdoor reviews are excluded from training.** This gives the sentiment model cleaner signal at the extremes, but means its real-world accuracy on genuinely mixed/ambiguous reviews is untested.
- **The Glassdoor scope is a documented allow-list** of ~25 named firms (Tech, Consulting, Finance), not the full 427 firms in the raw dataset. This was a deliberate choice to keep the analysis population comparable to a corporate, salaried workforce — stated explicitly here rather than left implicit.
- **The firm leaderboard only uses test-set predictions.** An earlier version of this pipeline accidentally scored firms using rows the model had already trained on, inflating their apparent accuracy. The current version only scores held-out test rows, and excludes firms with fewer than 50 test reviews as statistically unreliable.
- **Class imbalance is handled via `class_weight='balanced'`** in both models, not by discarding majority-class data — this preserves the full dataset size while still forcing the model to pay attention to the minority class.

---

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `statsmodels` (VIF) · `nltk` *(if used for any text preprocessing)* · `matplotlib` / `seaborn` (notebooks) · `plotly` (app) · `streamlit` · `joblib`
