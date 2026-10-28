# Weather forecasting evaluation — Ouagadougou 2015–2024

Experiments for the paper
*"A Comprehensive Evaluation of Weather Forecasting Techniques: A Case Study in Burkina Faso"*.

The project compares statistical, machine-learning and deep-learning approaches on ten years of
daily weather observations for Ouagadougou:

- **univariate forecasting** of daily temperature and precipitation (ARIMA, Prophet, stacked LSTM);
- **multivariate classification** of the daily weather condition (≈18 scikit-learn / boosting
  classifiers and a dense Keras network).

## Repository structure

```
.
├── Dataset/
│   └── data-01.csv … data-05.csv           # Visual Crossing daily data, 2 years per file
├── Model Univaries/
│   ├── arima_weather_forecasting_ouagadougou.ipynb
│   ├── prophet_weather_forecasting_ouagadougou.ipynb
│   └── stack_lstm_weather_forecasting_ouagadougou.ipynb
├── Model Multivaries/
│   └── Weather_Prediction_final_02.ipynb   # weather-condition classification
└── README.md
```

## Data

| | |
|---|---|
| Source | [Visual Crossing Weather](https://www.visualcrossing.com/) — Ouagadougou (stations `65503099999`, `DFFD`) |
| Period | 2015-01-01 → 2025-01-01, daily, no missing days |
| Rows | 3 658 rows, 3 654 distinct dates |
| Columns | 33 (temperature, feels-like, dew point, humidity, precipitation, wind, pressure, cloud cover, visibility, solar radiation, UV index, sunrise/sunset, `conditions`, …) |

Points to keep in mind when using the files:

- **Units are US/imperial**: temperatures in **°F**, precipitation in **inches**, wind in **mph**,
  visibility in **miles**. Convert before reporting results in °C / mm.
- **Overlapping dates**: consecutive files share their boundary day
  (2017-01-01, 2019-01-01, 2021-01-01, 2023-01-01). Some of these duplicates differ slightly
  (e.g. `visibility`, `severerisk` on 2023-01-01), so deduplicate on `datetime`, not on full rows.
- `snow` and `snowdepth` are always 0; `severerisk` is only available from 2023 onwards;
  `preciptype` is empty on dry days.
- The `conditions` label takes 5 values (`Partially cloudy`, `Rain, Partially cloudy`, `Clear`,
  `Overcast`, `Rain, Overcast`). Any `Rain` label corresponds exactly to `precip > 0`, so
  precipitation columns must not be used as features when predicting rain.

## Notebooks

### Univariate models — `Model Univaries/`

| Notebook | Target(s) | Resolution | Split |
|---|---|---|---|
| `arima_…` | `temp`, `precip` | monthly means | train 2015–2021, test 2021–2025 |
| `prophet_…` | `temp`, `precip` | daily | chronological 80 / 20 |
| `stack_lstm_…` | `temp`, `precip` | daily, 20-day input window | chronological 80 / 20 |

Metrics: MSE, RMSE, MAE, MAPE. MAPE is not meaningful for precipitation (many zero days).

### Multivariate classification — `Model Multivaries/`

`Weather_Prediction_final_02.ipynb` predicts a grouped weather label (`Clear`, `Partly Cloudy`,
`Rain`) from the daily measurements:

1. feature engineering — daylight duration, cyclical encoding of month / day-of-year / wind direction,
   temperature ranges, binary indicators;
2. median imputation and IQR winsorisation;
3. class balancing with Borderline-SMOTE;
4. feature selection (`feature_engine` + XGBoost gain importance, top 12 features);
5. training and comparison of ~18 classifiers, randomized hyper-parameter search for
   XGBoost / LightGBM / Random Forest, and a dense Keras network;
6. export of the best model (`best_model.pkl`, `scaler.pkl`, `label_encoder.pkl`,
   `feature_pipeline.pkl`, `weather_predictor.keras`).

## Getting started

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib seaborn plotly jupyter \
            statsmodels prophet scikit-learn tensorflow \
            xgboost lightgbm catboost imbalanced-learn feature_engine
# optional: pip install scikit-learn-intelex
```

The notebooks read `data-01.csv` … `data-05.csv` from their **working directory**. Either copy the
CSV files next to the notebook, or change the loading cell to point to the dataset folder, e.g.:

```python
import glob
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('../Dataset/data-0*.csv'))],
               ignore_index=True)
```

Then run the notebooks with `jupyter lab`.

## Evaluation protocol (reference)

The protocol below is the one intended for the results of the paper. Results produced by the
notebooks should be checked against it.

- **Data**: duplicate dates removed, units converted to SI on load.
- **Periods**: fitting 2015–2022 (2022 used as validation year for the neural networks),
  test 2023–2024.
- **Univariate models**: rolling-origin forecasts 1, 7 and 30 days ahead; every forecast for day *t*
  at horizon *h* uses observations up to *t − h* only. Persistence and daily climatology are
  reported as baselines.
- **Classification**: chronological split; all preprocessing (imputation, winsorisation, scaling,
  feature selection) fitted on the fitting period only; Borderline-SMOTE applied to training data
  only; the test set is made of real days; no feature derived from the target.
