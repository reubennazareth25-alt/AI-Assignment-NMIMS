# Pharma-Pulse: AI Demand Sensing Engine

Proof-of-Concept Streamlit app that compares **Traditional Planning** (Moving Average)
with an **AI Planning Engine** (Random Forest – LSTM ready) for a fictional basal
insulin brand: **InsuGuard Max 100 IU/mL** in India.

## What it demonstrates

- Synthetic but realistic daily **secondary demand** for InsuGuard Max
- External features:
  - Diabetes & Flu Google Trends Index (proxy for disease burden)
  - Price Index (WPI-like drift and trade schemes)
- Baseline model:
  - Moving Average forecast
- AI model:
  - Tree-based ML model using lagged sales + external signals
- **Concept/Data Drift**:
  - Kolmogorov–Smirnov (KS) test to detect distribution shift
  - Scenario toggle: Normal vs. Pandemic Shock (3× demand for 1 month)

## How to run locally

```bash
pip install -r requirements.txt
streamlit run pharma_pulse_app.py
