"""Pharma-Pulse: AI Demand Sensing Engine (Robust Version)

Streamlit Proof-of-Concept app that compares a traditional
moving-average planner with an AI model (LSTM-ready placeholder)
for pharma demand sensing, and includes a concept-drift demo
using a Kolmogorov–Smirnov (KS) test.

Run with:
    streamlit run pharma_pulse_app.py

Python 3.9+
Required packages (minimum):
    pip install streamlit pandas numpy scikit-learn scipy

Optional (if you want a real LSTM instead of the simple ML model):
    pip install tensorflow

This file is structured to map directly to the roles described:
    - Data Engineer: data loading + feature engineering
    - Baseline Analyst: moving-average baseline
    - AI Architect: AI model
    - MLOps Engineer: drift detection logic
    - Frontend/Business: Streamlit UI + narrative

This version adds extra checks so that it "just runs" even if:
    - The dataset is small
    - The moving-average window is too large
    - The lag value is too large
    - Uploaded data has messy types
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import streamlit as st

# -----------------------------------------------------------------------------
# 1. CONFIG & UTILITIES
# -----------------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)


@dataclass
class ForecastResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray

    @property
    def rmse(self) -> float:
        if len(self.y_true) == 0:
            return float("nan")
        return float(np.sqrt(mean_squared_error(self.y_true, self.y_pred)))


# -----------------------------------------------------------------------------
# 2. DATA ENGINEER: DATA LOADING & FEATURE ENGINEERING
# -----------------------------------------------------------------------------


def generate_synthetic_pharma_data(n_days: int = 730) -> pd.DataFrame:
    """Generate synthetic daily insulin sales data with seasonality
    and an explicit "flu season" spike.

    Columns:
        date, sales, google_trends_flu, price_index
    """

    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")

    # Base demand around 100 units with weekly pattern
    base_demand = 100 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)

    # Seasonality: higher in winter months (Nov–Feb)
    months = dates.month
    winter_boost = np.where(np.isin(months, [11, 12, 1, 2]), 25, 0)

    # Flu season spike window (e.g., Dec–Jan)
    flu_spike = np.where(np.isin(months, [12, 1]), 40, 0)

    noise = np.random.normal(0, 8, size=n_days)

    sales = base_demand + winter_boost + flu_spike + noise
    sales = np.clip(sales, a_min=30, a_max=None)

    # External Feature 1: Google Flu Trends – correlated with sales but noisy
    google_trends_flu = (
        (sales - sales.min()) / max(sales.max() - sales.min(), 1e-9) * 80
        + np.random.normal(0, 5, size=n_days)
    )

    # External Feature 2: Price Index – slow random walk
    price_index = 100 + np.cumsum(np.random.normal(0, 0.05, size=n_days))

    df = pd.DataFrame(
        {
            "date": dates,
            "sales": sales,
            "google_trends_flu": google_trends_flu,
            "price_index": price_index,
        }
    )

    return df


@st.cache_data(show_spinner=False)
def load_data(use_uploaded: bool, uploaded_file) -> pd.DataFrame:
    """Load real data from upload or fall back to synthetic.

    Expected columns for real data: date, sales.
    The function will add synthetic external features even for real data,
    to keep the demo consistent.
    """

    if use_uploaded and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Attempt to normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        if "date" not in df.columns or "sales" not in df.columns:
            raise ValueError("Uploaded CSV must have at least 'date' and 'sales' columns.")

        # Parse date and sort
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Coerce sales to numeric
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
        df = df.dropna(subset=["sales"]).reset_index(drop=True)

        if len(df) < 60:
            # If the uploaded data is too small, just fall back to synthetic
            return generate_synthetic_pharma_data(n_days=730)

        # If no external features, create synthetic ones based on sales
        if "google_trends_flu" not in df.columns:
            sales = df["sales"].values
            denom = max(sales.max() - sales.min(), 1e-9)
            scaled = (sales - sales.min()) / denom
            df["google_trends_flu"] = scaled * 80 + np.random.normal(0, 5, size=len(df))

        if "price_index" not in df.columns:
            df["price_index"] = 100 + np.cumsum(np.random.normal(0, 0.05, size=len(df)))

        return df

    # Fallback: synthetic pharma data
    return generate_synthetic_pharma_data(n_days=730)


# -----------------------------------------------------------------------------
# 3. FEATURE ENGINEERING FOR TIME-SERIES MODELS
# -----------------------------------------------------------------------------


def create_lagged_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    n_lags: int = 30,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Turn a univariate + external-features series into a supervised
    learning dataset with lagged features.

    X: lagged sales + google_trends_flu + price_index
    y: current sales
    """

    data = df.copy().reset_index(drop=True)

    # Cap n_lags to length of data - 2 (at least a couple of rows left)
    n_lags = int(min(max(n_lags, 1), max(len(data) - 2, 1)))

    # Create lag features for target
    for lag in range(1, n_lags + 1):
        data[f"{target_col}_lag_{lag}"] = data[target_col].shift(lag)

    # Drop the first n_lags rows (they have NaNs in lag columns)
    data = data.dropna().reset_index(drop=True)

    if len(data) == 0:
        raise ValueError("Not enough data after creating lagged features. Try reducing n_lags.")

    feature_cols = [
        col
        for col in data.columns
        if col.startswith(f"{target_col}_lag_")
        or col in ["google_trends_flu", "price_index"]
    ]

    X = data[feature_cols]
    y = data[target_col]

    return X, y


# -----------------------------------------------------------------------------
# 4. BASELINE ANALYST: TRADITIONAL MOVING-AVERAGE MODEL
# -----------------------------------------------------------------------------


def moving_average_forecast(series: pd.Series, window: int = 30) -> np.ndarray:
    """Simple moving-average forecast.

    For each time t in the train period, prediction = average of
    previous `window` observations.
    """

    values = series.values
    # Ensure window is valid
    window = int(min(max(window, 1), max(len(values) - 1, 1)))

    preds = []
    for t in range(window, len(values)):
        preds.append(np.mean(values[t - window : t]))

    # Align lengths: first `window` points can't be predicted
    return np.array(preds)


# -----------------------------------------------------------------------------
# 5. AI ARCHITECT: SIMPLE ML MODEL (LSTM-READY PLACEHOLDER)
# -----------------------------------------------------------------------------


def train_ai_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestRegressor:
    """Train a simple tree-based model as an AI baseline.

    For a more advanced version, replace this with an LSTM implemented
    in TensorFlow/Keras. The interface (fit/predict) can remain similar
    from the Streamlit app's point of view.
    """

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------------------------------------------------------
# 6. MLOPS ENGINEER: DATA / CONCEPT DRIFT DETECTION
# -----------------------------------------------------------------------------


def check_drift(reference: np.ndarray, current: np.ndarray, threshold: float = 0.05):
    """Kolmogorov–Smirnov test for distribution shift.

    Returns (drift_detected, p_value).
    """

    if len(reference) == 0 or len(current) == 0:
        return False, 1.0

    statistic, p_value = ks_2samp(reference, current)
    drift_detected = p_value < threshold
    return drift_detected, p_value


# -----------------------------------------------------------------------------
# 7. STREAMLIT FRONTEND / BUSINESS NARRATIVE
# -----------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Pharma-Pulse: AI Demand Sensing Engine",
        layout="wide",
    )

    st.title("💊 Pharma-Pulse: AI Demand Sensing Engine")
    st.markdown(
        """
        **Case Study: InsuGuard Max – Basal Insulin Supply Chain in India**

        This Proof-of-Concept simulates the demand and supply planning for a
        fictional but realistic basal insulin brand, **InsuGuard Max 100 IU/mL**
        (10 mL vial), marketed across **Metro + Tier-II cities in India**.

        We compare **Traditional Planning** (Moving Average) with an
        **AI Planning Engine** (tree-based model, LSTM-ready) to manage:

        - Volatile demand during **flu / viral seasons** when glycaemic control worsens.
        - **Prescription surges** from Diabetologists & Physicians.
        - **Price changes** and their impact on primary sales.

        We also demonstrate **Data / Concept Drift Detection** using a
        Kolmogorov–Smirnov (KS) test when a **Black Swan event** like
        COVID-style panic buying hits the market.
        """
    )

    st.markdown(
        """
        ### 🧬 Brand & Supply Chain Narrative

        - **SKU**: InsuGuard Max 100 IU/mL Vial, 10 mL (Cold-chain, Rx only).
        - **Manufacturing Site**: Goa Formulations Plant – single primary site
          with a **60–75 day end-to-end lead time** from API to CFA.
        - **Network**: 1 Central Depot → 4 CFAs (North, South, East, West) → ~80 Super Stockists → ~8,000 Retail Pharmacies.
        - **Planning Cadence**: Monthly rolling forecast, weekly replenishment.
        - **Business KPIs**: OTIF ≥ 95%, Stock-out days < 2/month per CFA, Expiry loss < 0.5% of NSV.

        The synthetic dataset you see below represents **daily secondary demand**
        (pharmacy off-takes) for InsuGuard Max, influenced by:

        - **Seasonality** (higher in winter months when infections increase).
        - A sharp **flu-season spike** (Dec–Jan) where insulin requirements rise.
        - A noisy but correlated **"Diabetes & Flu Google Trends Index"** used as
          a proxy for disease activity.
        - A slow-moving **Price Index** reflecting incremental WPI-linked price
          increases and trade schemes.
        """
    )

    # Sidebar: data selection & parameters
    st.sidebar.header("Data & Scenario Setup")

    st.sidebar.markdown("### 1. Data Source")
    use_uploaded = st.sidebar.checkbox("Use uploaded CSV instead of synthetic data")
    uploaded_file = None
    if use_uploaded:
        uploaded_file = st.sidebar.file_uploader("Upload CSV (date, sales, ...)", type=["csv"])

    window = st.sidebar.slider("Moving Average Window (days)", min_value=7, max_value=60, value=30, step=1)

    n_lags = st.sidebar.slider("AI Model: Number of Sales Lags", min_value=7, max_value=60, value=30, step=1)

    test_fraction = st.sidebar.slider("Test Set Fraction", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 2. Drift Scenario")
    drift_scenario = st.sidebar.radio(
        "Select Market Scenario",
        ("Normal Operations", "Pandemic Shock (3x Demand for 1 Month)"),
    )

    # Load data
    try:
        df = load_data(use_uploaded, uploaded_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    if df is None or len(df) < 60:
        st.error("Dataset too small to build a meaningful demo. Please provide more rows.")
        return

    df = df.sort_values("date").reset_index(drop=True)

    st.subheader("Phase A: Data Overview")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Daily Sales (Synthetic or Uploaded)**")
        st.line_chart(df.set_index("date")["sales"])

    with col2:
        st.write("Sample of the dataset:")
        st.dataframe(df.head(10))

    # ------------------------------------------------------------------
    # Train/Test Split
    # ------------------------------------------------------------------
    n = len(df)
    test_size = int(n * test_fraction)
    test_size = max(min(test_size, n // 2), 10)  # keep it reasonable

    if n - test_size <= 30:
        test_size = max(10, n // 3)

    train_df = df.iloc[: n - test_size].reset_index(drop=True)
    test_df = df.iloc[n - test_size :].reset_index(drop=True)

    # Baseline: Moving Average
    baseline_series = train_df["sales"]
    baseline_preds = moving_average_forecast(baseline_series, window=window)

    if len(baseline_preds) == 0:
        st.warning("Moving-average window too large for the training data. Reducing window automatically.")
        baseline_preds = moving_average_forecast(baseline_series, window=7)

    y_true_baseline = baseline_series.values[len(baseline_series) - len(baseline_preds) :]

    baseline_result = ForecastResult(
        name=f"Moving Average ({window}d)",
        y_true=y_true_baseline,
        y_pred=baseline_preds,
    )

    # AI Model Data
    try:
        X, y = create_lagged_features(df, target_col="sales", n_lags=n_lags)
    except ValueError as e:
        st.error(str(e))
        return

    n2 = len(X)
    test_size2 = int(n2 * test_fraction)
    test_size2 = max(min(test_size2, n2 // 2), 10)

    if n2 - test_size2 <= 10:
        test_size2 = max(5, n2 // 3)

    X_train, X_test = X.iloc[: n2 - test_size2], X.iloc[n2 - test_size2 :]
    y_train, y_test = y.iloc[: n2 - test_size2], y.iloc[n2 - test_size2 :]

    if len(X_train) == 0 or len(X_test) == 0:
        st.error("Not enough data to train/test the AI model. Try reducing lags or test fraction.")
        return

    model = train_ai_model(X_train, y_train)
    ai_preds = model.predict(X_test)

    ai_result = ForecastResult(
        name="AI Model (Random Forest – LSTM ready)",
        y_true=y_test.values,
        y_pred=ai_preds,
    )

    # ------------------------------------------------------------------
    # Phase B vs Phase C: Traditional vs AI
    # ------------------------------------------------------------------
    st.subheader("Phase B & C: Traditional vs AI Planning")

    col_ma, col_ai = st.columns(2)

    with col_ma:
        st.markdown("#### Traditional Planning – Moving Average")
        st.write(
            "The moving-average forecast reacts **slowly** to sudden demand spikes,"
            " causing stock-outs when the market changes abruptly."
        )
        st.metric("RMSE (Train Period)", f"{baseline_result.rmse:.2f}")

    with col_ai:
        st.markdown("#### AI Planning – Multivariate Model")
        st.write(
            "The AI engine uses **lagged sales + external signals** (flu trends, price index)"
            " to respond earlier to spikes, reducing forecast error."
        )
        st.metric("RMSE (Hold-out Test)", f"{ai_result.rmse:.2f}")

    st.markdown("**Forecast Comparison (Hold-out Test Window)**")

    # Build a comparison DataFrame for plotting
    test_dates = df["date"].iloc[-len(y_test) :].reset_index(drop=True)
    comp_df = pd.DataFrame(
        {
            "date": test_dates,
            "Actual_Sales": y_test.values,
            "AI_Forecast": ai_preds,
        }
    )
    comp_df = comp_df.set_index("date")

    st.line_chart(comp_df)

    st.caption(
        "Notice how the **AI forecast** line tends to hug the actual demand "
        "more closely than a naive moving average, especially around peaks."
    )

    # ------------------------------------------------------------------
    # Phase D: Drift Detection – Black Swan shock
    # ------------------------------------------------------------------
    st.subheader("Phase D: Model Decay & Data Drift Detection")

    st.markdown(
        """
        We now simulate a **Black Swan event** (e.g., COVID-19 panic buying)
        by injecting a sudden, persistent change in demand.

        The system monitors the distribution of recent demand against
        the historical training distribution using a **KS Test**.
        """
    )

    # Reference distribution: training sales
    reference = train_df["sales"].values

    # Current data window: recent demand (copy from test_df)
    current_df = test_df.copy()

    if drift_scenario == "Pandemic Shock (3x Demand for 1 Month)":
        # Multiply last 30 days by 3 to emulate panic buying
        shock_window = min(30, len(current_df))
        if shock_window > 0:
            idx_to_change = current_df.index[-shock_window:]
            current_df.loc[idx_to_change, "sales"] = current_df.loc[idx_to_change, "sales"] * 3

    current = current_df["sales"].values

    drift_detected, p_val = check_drift(reference, current, threshold=0.05)

    col_plot, col_msg = st.columns([2, 1])

    with col_plot:
        st.markdown("**Recent Demand (Monitored Window)**")
        st.line_chart(current_df.set_index("date")["sales"])

    with col_msg:
        if drift_detected:
            st.error(
                f"🚨 CRITICAL ALERT: Data / Concept Drift Detected (p-value = {p_val:.4f})"
            )
            st.markdown(
                "**Recommended Actions:**\n"
                "- Freeze current model for high-risk SKUs.\n"
                "- Trigger automated **retraining pipeline** on latest data.\n"
                "- Notify Supply Chain & Brand teams to review safety stocks."
            )
        else:
            st.success(
                f"✅ Model Health: Stable (p-value = {p_val:.4f}). No significant drift detected."
            )
            st.markdown(
                "The recent demand distribution is statistically consistent with the"
                " training data. No immediate retraining action required."
            )

    st.markdown("---")
    st.markdown("### Architecture & Talking Points for Presentation")

    st.markdown(
        """
        **Slide 1 – Business Problem**  

        - Pharma inventory is rigid. We lose money on **expiry** and **stock-outs**.
        - Volatile demand during **flu season** or **pandemics** breaks traditional planning.

        **Slide 2 – Solution Overview**  

        - Move from **Univariate Forecasting** (only past sales) to **Multivariate AI Sensing**.  
        - Inputs: **Past Sales + Google Flu Index + Price Index**.  
        - Output: **Daily demand forecast** for critical molecules (e.g., Insulin).

        **Slide 3 – Technical Architecture**  

        - Data Source (SQL / CSV / API) → **Python/Pandas** for feature engineering.  
        - Models:  
            - Baseline: **Moving Average / ARIMA**.  
            - AI: **Tree-based model / LSTM** for multivariate time-series.  
        - Monitoring: **KS-test based drift detector**.  
        - UI: **Streamlit Micro-App** for planners & management.

        **Slide 4 – Evaluation (RMSE Table)**  

        - Show RMSE of **ARIMA / Moving Average vs. AI Model**.  
        - Highlight lower error of AI especially around **spike periods**.

        **Slide 5 – Drift & Governance**  

        - Normal vs. Pandemic Shock scenario.  
        - Drift alert: "Distribution Shift > Threshold → Retraining Recommended".  
        - Link to **Model Risk Governance** and **Lifecycle Management**.
        """
    )

    st.info(
        "To convert the current Random Forest placeholder into a true LSTM, "
        "swap `train_ai_model` with a Keras implementation that consumes the "
        "same lagged features as a sequence. The rest of the app and narrative "
        "remains identical for the professor."
    )


if __name__ == "__main__":
    main()
