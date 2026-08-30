# Does XGBoost Actually Beat Linear Regression? (A Used Car Price Case Study)

A hands-on investigation into when gradient boosting earns its complexity, and when it doesn't.

## The setup

Predicting used car `selling_price` from the [CarDekho used car dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/car-details-dataset) (4,340 listings: year, km driven, fuel type, transmission, seller type, ownership history). The plan: train Linear Regression and XGBoost on the same data, compare, and understand why one wins.

## Getting the data model-ready

Raw categorical columns need encoding before either model can use them, and the encoding choice matters:

- **`owner`** (First Owner, Second Owner, etc.) has a genuine order, so it's label-encoded to a single numeric column (0 to 4).
- **`fuel`, `seller_type`, `transmission`, `brand`** have no real order, so they're one-hot encoded with `pd.get_dummies(..., drop_first=True)`. Dropping one category per column avoids the dummy variable trap: keeping all categories creates a column that's a perfect linear combination of the others (e.g. `fuel_CNG = 1 - fuel_Petrol - fuel_Diesel`), which breaks linear regression's coefficient estimation.
- **`brand`** was extracted from the free-text `name` column (e.g. "Maruti 800 AC" to "Maruti"), but only added in Round 3 below. Rare brands (under 15 listings) were bucketed into `Other` to avoid one-hot columns with almost no data to learn from.

## Round 1: a surprisingly close fight

| Model | Test RMSE |
|---|---|
| Linear Regression | ₹4.68L |
| XGBoost (default params) | ₹4.44L |

Only about 6% better. Not the blowout I expected. XGBoost is supposed to crush linear models on nonlinear data, so either the data was too simple, or the model wasn't tuned.

## Round 2: tuning, and a bug that almost fooled me

Tuning `learning_rate`, `n_estimators`, `max_depth`, and subsampling first appeared to drop RMSE to a dramatic **₹2.61L**, a 40%+ improvement. Turned out the comparison was computing `RMSE(tuned_predictions, linear_regression_predictions)` instead of `RMSE(tuned_predictions, true_values)`, comparing two prediction arrays against each other instead of against the actual target. A basic but easy mistake to make mid-notebook.

Rerunning clean, with `early_stopping_rounds` letting a validation split (not guesswork) pick the right number of trees:

| Model | Test RMSE |
|---|---|
| Linear Regression | ₹4.68L |
| XGBoost (default) | ₹4.44L |
| XGBoost (tuned, early-stopped) | ₹4.35L |

The real gap: about 7%. Meaningful, but modest, and a good reminder to always check a prediction against the true target, never against another model's predictions.

## Round 3: adding `brand`, and hitting a tuning ceiling

The dataset had a `name` column that I'd initially dropped instead of mining for `brand`, arguably the single strongest price signal in used cars, and a textbook nonlinear, non-ordinal feature that trees are built to exploit. After extracting and encoding it:

| Model | Test RMSE |
|---|---|
| Linear Regression (+brand) | ₹4.03L |
| XGBoost default (+brand) | ₹3.62L |

Adding `brand` closed far more of the gap than tuning did in Round 2: XGBoost's default configuration alone dropped from ₹4.44L to ₹3.62L, an 18% improvement, just from one better-engineered feature.

Naturally the next question was whether tuning could push this further. A full grid search (5-fold CV over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`) found a best configuration scoring **₹3.65L on the test set, statistically the same as the untouched default's ₹3.62L.** Once `brand` supplied real signal, tuning had nothing left to add. The default model is used going forward, since it's simplest and ties the tuned result.

## Proving the "why" with SHAP interaction values

Theory: trees can learn `brand x year` interactions (different brands depreciate on different curves) automatically, with zero manual feature engineering, something linear regression structurally cannot do without an explicit interaction term. Verified with SHAP TreeExplainer interaction values on the final default (+brand) model:

- **Maruti**, **Mercedes-Benz**, **Toyota**, **Tata**, **BMW** showed by far the strongest brand x year interaction strength (roughly 3,900 to 6,500)
- **Nissan**, **Fiat** showed the weakest (under 700, several times lower than the top brands)

This lines up with real-world intuition: luxury brands (BMW, Mercedes) hold value on a different curve than mass-market brands, and XGBoost picked that up on its own, no manual interaction term needed.

## Takeaways

- Feature engineering (`brand`) moved the needle far more than hyperparameter tuning did, 18% from one feature versus 7% from tuning. A full grid search after adding `brand` found nothing left to improve.
- XGBoost's advantage over linear regression is proportional to how much real nonlinearity and interaction exists in the data. It's not a free win by default, and it's not unlimited either: once the model has the signal it needs, more tuning doesn't automatically mean more accuracy.
- Always sanity-check dramatic improvements. A prediction-vs-prediction comparison nearly turned a real 7% gain into a fake 40% headline.
- `feature_importances_` type matters: `gain` alone suggested `transmission` was the dominant feature; checking `weight` and `cover` alongside it told a more trustworthy story.

## Stack

`pandas`, `scikit-learn`, `xgboost`, `shap`

---
*Dataset: [CarDekho Used Car Data](https://www.kaggle.com/datasets/akshaydattatraykhare/car-details-dataset), Kaggle*
