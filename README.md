# ⚽ Premier League Match Outcome Prediction with Random Forest

This project uses historical Premier League match data to build a machine learning model that predicts match outcomes based on rolling team statistics and match context. The model is trained using a **Random Forest Classifier** and includes a method for comparing predicted outcomes between opposing teams in the same match.

---

## 📁 Files Included

- `PLPredictorV1.ipynb`: Jupyter notebook with the full data pipeline, model training, evaluation, and analysis.
- `matches.csv`: Cleaned dataset of Premier League matches containing date, team, opponent, result, and match statistics.

---

## 🔍 Key Features

- **Data Preprocessing:**
  - Time and date formatting
  - Encoding categorical variables (venue, opponent, etc.)
  - Target variable creation (`1` for win, `0` for non-win)

- **Rolling Feature Engineering:**
  - Computation of 3-game rolling averages for key stats like goals for/against, shots, distance, penalties, etc.
  - Grouping by team and sorting chronologically to ensure realistic feature generation

- **Model Training:**
  - Train/test split based on a cutoff date (`2022-01-01`)
  - Random Forest Classifier trained on game context and rolling stats

- **Prediction Comparison:**
  - Merge model predictions for both teams in the same game
  - Analyze prediction symmetry/conflict (e.g., one model predicts win, other predicts loss)

---

## 📊 Model Performance

- **Accuracy Score:** Evaluated on post-2022 matches
- **Precision Score:** Measures how many predicted wins were actually wins
- **Crosstab Analysis:** Shows distribution of actual vs. predicted outcomes

---

## 📈 Example Use Case

Identify games where the model predicts a win for one team and a loss for the other, and check how often the model gets it right:

```python
merged[(merged["predictions_x"] == 1) & (merged["predictions_y"] == 0)]["actual_x"].value_counts()
