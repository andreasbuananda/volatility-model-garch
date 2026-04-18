# Prognose der finanziellen Volatilität der Bitcoin mittels GARCH-Modell
# Forecasting Bitcoin Financial Volatility using GARCH Models

---

## 🇩🇪 Deutsch

### Projektbeschreibung

Dieses Repository enthält den vollständigen Code und die Notebooks für eine Bachelorarbeit zum Thema:

**„Prognose der finanziellen Volatilität der Bitcoin mittels GARCH-Modell"**

Ziel ist es, die Volatilität des Bitcoin/US-Dollar-Kurses (BTC-USD) mithilfe von GARCH-Familienmodellen zu modellieren und zu prognostizieren. Die Analyse folgt einer strengen ökonometrischen Pipeline: von der explorativen Datenanalyse über Stationaritätstests bis hin zur Modellschätzung, Diagnose und Prognose.

### Forschungsfragen

1. Erfasst das GARCH(1,1)-Modell die Volatilitätsdynamik von Bitcoin-Renditen angemessen?
2. Welche GARCH-Variante (GARCH, EGARCH, GJR-GARCH) liefert die besten Prognoseergebnisse für Bitcoin?
3. Welche Fehlerverteilung (Normal, Student-t, Skewed-t) passt am besten zu den Fat-Tail-Eigenschaften von Bitcoin-Renditen?

### Methodik

```
1. Explorative Datenanalyse (EDA)
        ↓
2. Stationaritätstests (ADF, KPSS)
        ↓
3. ARMA-Mittelwertmodell-Selektion
        ↓
4. ARCH-LM-Test (Nachweis von ARCH-Effekten)
        ↓
5. GARCH-Modell-Schätzung (GARCH, EGARCH, GJR-GARCH)
        ↓
6. Residualdiagnose (Ljung-Box, ACF/PACF)
        ↓
7. Volatilitätsprognose & Evaluation (RMSE, MAE, QLIKE)
```

---

## 🇬🇧 English

### Project Description

This repository contains the complete code and notebooks for a bachelor's thesis on:

**"Forecasting Bitcoin Financial Volatility using GARCH Models"**

The goal is to model and forecast the volatility of Bitcoin/USD (BTC-USD) exchange rates using GARCH-family models. The analysis follows a rigorous econometric pipeline: from exploratory data analysis and stationarity testing to model estimation, diagnostics, and forecasting.

### Research Questions

1. Does the GARCH(1,1) model adequately capture the volatility dynamics of Bitcoin returns?
2. Which GARCH variant (GARCH, EGARCH, GJR-GARCH) delivers the best forecast performance for Bitcoin?
3. Which error distribution (Normal, Student-t, Skewed-t) best fits the fat-tail properties of Bitcoin returns?

### Methodology

```
1. Exploratory Data Analysis (EDA)
        ↓
2. Stationarity Tests (ADF, KPSS)
        ↓
3. ARMA Mean Model Selection
        ↓
4. ARCH-LM Test (detecting ARCH effects)
        ↓
5. GARCH Model Estimation (GARCH, EGARCH, GJR-GARCH)
        ↓
6. Residual Diagnostics (Ljung-Box, ACF/PACF)
        ↓
7. Volatility Forecast & Evaluation (RMSE, MAE, QLIKE)
```

### Repository Structure

```
bitcoin-volatility-garch/
├── .github/
│   └── copilot-instructions.md   # Copilot session context
├── data/
│   ├── raw/                      # Raw BTC/USD price data (CSV)
│   └── processed/                # Cleaned and engineered data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_stationarity_tests.ipynb
│   ├── 03_arma_mean_model.ipynb
│   ├── 04_garch_modeling.ipynb
│   ├── 05_model_diagnostics.ipynb
│   └── 06_volatility_forecast.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Fetch BTC data via yfinance
│   ├── preprocessing.py          # Log returns, stationarity, outliers
│   ├── model_builder.py          # GARCH model factory
│   ├── diagnostics.py            # Ljung-Box, ARCH-LM, ACF/PACF
│   └── forecast.py               # Rolling and static forecasting
├── tests/
│   └── test_preprocessing.py
├── results/
│   ├── figures/
│   └── tables/
├── requirements.txt
├── README.md
└── .gitignore
```

### Installation

```bash
# Clone the repository
git clone https://github.com/andreasbuananda/volatility-model-garch.git
cd volatility-model-garch

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

Execute notebooks in order for a complete analysis pipeline:

```bash
jupyter lab
```

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Download BTC data, visualize prices and returns |
| `02_stationarity_tests.ipynb` | ADF and KPSS tests, differencing |
| `03_arma_mean_model.ipynb` | ACF/PACF, ARMA model selection via AIC/BIC |
| `04_garch_modeling.ipynb` | Fit GARCH, EGARCH, GJR-GARCH models |
| `05_model_diagnostics.ipynb` | Ljung-Box, ARCH-LM residual diagnostics |
| `06_volatility_forecast.ipynb` | Rolling forecast, RMSE/MAE/QLIKE evaluation |

### Running Tests

```bash
pytest tests/ -v
```

### Key References

- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity*. Journal of Econometrics, 31(3), 307–327.
- Engle, R. F. (1982). *Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation*. Econometrica, 50(4), 987–1007.
- Ardia, D., Bluteau, K., Boudt, K., Catania, L., & Trottier, D.-A. (2019). *Markov-Switching GARCH Models in R: The MSGARCH Package*. Journal of Statistical Software, 91(4), 1–38.
- Nelson, D. B. (1991). *Conditional Heteroskedasticity in Asset Returns: A New Approach*. Econometrica, 59(2), 347–370.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). *On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks*. Journal of Finance, 48(5), 1779–1801.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
