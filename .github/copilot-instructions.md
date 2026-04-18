# Copilot Instructions — Bitcoin Volatility GARCH Project

## Project Summary

This is a Python research repository for a bachelor's thesis titled:
**"Prognose der finanziellen Volatilität der Bitcoin mittels GARCH-Modell"**
(Forecasting Bitcoin Financial Volatility using GARCH Models)

The project models and forecasts Bitcoin/USD (BTC-USD) price volatility using
GARCH-family models (GARCH, EGARCH, GJR-GARCH) with multiple error distributions.

---

## Tech Stack

| Component | Library / Tool |
|-----------|---------------|
| Language | Python 3.10+ |
| GARCH Modeling | `arch` (>=6.2.0) |
| Statistical Tests | `statsmodels` (ADF, KPSS, Ljung-Box, ARCH-LM) |
| Data Fetching | `yfinance` (BTC-USD daily OHLCV) |
| Data Manipulation | `pandas`, `numpy`, `scipy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| ML Metrics | `scikit-learn` (RMSE, MAE) |
| Custom QLIKE | Implemented in `src/forecast.py` |
| Notebooks | `jupyter`, `jupyterlab` |
| Testing | `pytest` in `tests/` folder |

---

## Repository Layout

```
src/
├── __init__.py          # Package exports
├── data_loader.py       # Download & save BTC data via yfinance
├── preprocessing.py     # Log returns, stationarity, outlier handling
├── model_builder.py     # GARCH/EGARCH/GJR-GARCH model factory
├── diagnostics.py       # Ljung-Box, ARCH-LM, ACF/PACF plots
└── forecast.py          # Rolling and static forecasting + metrics

notebooks/               # Ordered analysis pipeline
tests/                   # pytest unit tests
data/raw/                # Raw CSV from yfinance (gitignored)
data/processed/          # Processed CSV with log_return col (gitignored)
results/figures/         # Generated plots (gitignored)
results/tables/          # Generated tables (gitignored)
```

---

## Coding Style

- **PEP 8**: All code must comply. Run `flake8 src/ tests/` to check.
- **Type hints**: All function signatures must include type hints.
- **Docstrings**: Use NumPy-style docstrings for all public functions and classes.
- **File paths**: Always use `pathlib.Path` — **never** `os.path` or hardcoded strings.
- **Imports**: Group as: stdlib → third-party → local. Separate with blank lines.
- **No magic numbers**: Define constants at the module level.

---

## Key Domain Knowledge

### Log Returns
```python
r_t = log(P_t / P_{t-1})
```
- Use log returns, NOT percentage returns, for time-additivity.
- Returns are typically scaled ×100 before GARCH estimation.

### GARCH(1,1) Model
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```
- **Stationarity constraint**: α + β < 1
- **Persistence**: α + β close to 1 means high volatility clustering.
- ω > 0, α ≥ 0, β ≥ 0 required.

### EGARCH (Nelson, 1991)
```
log(σ²_t) = ω + α·(|z_{t-1}| - E|z_{t-1}|) + γ·z_{t-1} + β·log(σ²_{t-1})
```
- γ < 0 captures leverage effect (bad news increases volatility more than good news).
- No non-negativity constraints needed (log variance).

### GJR-GARCH (Glosten-Jagannathan-Runkle, 1993)
```
σ²_t = ω + (α + γ·I_{t-1})·ε²_{t-1} + β·σ²_{t-1}
```
- I_{t-1} = 1 if ε_{t-1} < 0 (leverage indicator)
- γ > 0 means negative shocks increase volatility more.

### Bitcoin Return Stylized Facts
- **Volatility clustering**: Large moves tend to be followed by large moves.
- **Fat tails / leptokurtosis**: Returns have heavier tails than Normal — use Student-t or Skewed-t.
- **Leverage effect**: Negative returns increase future volatility more than positive returns.

### Statistical Tests
| Test | H0 | Reject H0 if |
|------|----|-------------|
| ADF | Unit root (non-stationary) | p < α → stationary |
| KPSS | Stationary | p < α → non-stationary |
| Ljung-Box | No autocorrelation | p < α → autocorrelation present |
| ARCH-LM | No ARCH effects | p < α → ARCH effects present |

### Evaluation Metrics
- **RMSE**: `sqrt(mean((actual - predicted)²))` — penalizes large errors.
- **MAE**: `mean(|actual - predicted|)` — robust to outliers.
- **QLIKE**: `mean(log(σ̂²) + r²/σ̂²)` — theoretically motivated, scale-invariant.

---

## Analysis Pipeline (strict order)

1. **EDA** (`01_data_exploration.ipynb`) — visualize prices, returns, distributions.
2. **Stationarity** (`02_stationarity_tests.ipynb`) — ADF + KPSS tests.
3. **ARMA Mean Model** (`03_arma_mean_model.ipynb`) — ACF/PACF, AIC/BIC selection.
4. **ARCH-LM Test** (in `04_garch_modeling.ipynb`) — confirm ARCH effects.
5. **GARCH Fitting** (`04_garch_modeling.ipynb`) — GARCH, EGARCH, GJR-GARCH.
6. **Diagnostics** (`05_model_diagnostics.ipynb`) — Ljung-Box on standardized residuals.
7. **Forecast & Evaluation** (`06_volatility_forecast.ipynb`) — rolling forecast, RMSE/MAE/QLIKE.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

Test files should mirror the module structure:
- `tests/test_preprocessing.py` → tests for `src/preprocessing.py`
- Add `tests/test_model_builder.py`, `tests/test_forecast.py` as needed.

---

## Common Copilot Prompts

- *"Fit a GARCH(1,1) model with Student-t distribution to BTC returns"*
- *"Plot the conditional volatility alongside BTC price"*
- *"Run the full diagnostic battery on the fitted model"*
- *"Perform rolling forecast evaluation and compare models by RMSE"*
- *"Test for leverage effects using GJR-GARCH and EGARCH"*
