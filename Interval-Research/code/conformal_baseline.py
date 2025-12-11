"""
Split Conformal Prediction Baseline - TRUE Implementation
==========================================================

Implements proper split conformal prediction for time series intervals.
This uses actual forecast residuals (actual - forecast) computed on a
held-out calibration set, NOT level-based residuals.

Key difference from empirical method:
- Empirical: residuals = y - mean(y)  [level deviations]
- Conformal: residuals = actual - forecast [forecast errors]

References:
- Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.
- Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized Quantile Regression.
"""

import numpy as np
from typing import Dict, List, Optional


# Standard 23-quantile grid
QGRID_23 = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99
]


def _qname(q: float) -> str:
    """Convert quantile level to column name (matching empirical method)."""
    if q == 0.01: return "q0_01"
    if q == 0.025: return "q0_025"
    if q == 0.05: return "q0_05"
    if q == 0.975: return "q0_975"
    if q == 0.99: return "q0_99"
    return f"q0_{int(round(q*100)):02d}"


def split_conformal_intervals(
    y: np.ndarray,
    point_forecast: float,
    horizon: int,
    cal_fraction: float = 0.3,
    min_cal_size: int = 20
) -> Dict[str, float]:
    """
    Generate prediction intervals using TRUE split conformal prediction.

    This computes actual forecast residuals on a calibration set:
    1. Split data: training set | calibration set
    2. For each point in calibration set, compute h-step-ahead forecast error
    3. Use empirical distribution of these errors for intervals

    Args:
        y: Historical time series
        point_forecast: Point forecast (persistence) for target
        horizon: Forecast horizon
        cal_fraction: Fraction of data for calibration (0.3 = 30%)
        min_cal_size: Minimum calibration set size

    Returns:
        Dict with quantile values using same naming as empirical method
    """
    n = len(y)

    # Determine calibration set size
    cal_size = max(min_cal_size, int(n * cal_fraction))
    train_size = n - cal_size

    if train_size < horizon + 5:
        # Fallback: use all available data
        train_size = max(horizon + 5, n // 2)
        cal_size = n - train_size

    # Compute forecast errors on calibration set
    # Using persistence forecast: prediction at t+h is y[t]
    forecast_errors = []

    for t in range(train_size, n - horizon):
        # Persistence forecast made at time t for t+horizon
        forecast = y[t]
        actual = y[t + horizon]
        error = actual - forecast
        forecast_errors.append(error)

    if len(forecast_errors) < 5:
        # Fallback: use more of the series
        forecast_errors = []
        for t in range(horizon, n - horizon):
            forecast = y[t]
            actual = y[t + horizon]
            forecast_errors.append(actual - forecast)

    if len(forecast_errors) < 3:
        # Ultimate fallback for very short series
        forecast_errors = np.diff(y[-min(20, len(y)):]).tolist()

    errors = np.array(forecast_errors)
    n_cal = len(errors)

    # Apply horizon scaling (same as empirical for fair comparison)
    scale = 1.0 + 0.1 * horizon

    # Generate quantiles with conformal finite-sample correction
    quantiles = {}
    for q in QGRID_23:
        # Conformal adjustment: use (n+1)*q / n quantile level
        # This provides finite-sample coverage guarantee
        if q <= 0.5:
            q_adj = max(0.001, (q * (n_cal + 1) - 1) / n_cal)
        else:
            q_adj = min(0.999, (q * (n_cal + 1)) / n_cal)

        q_val = np.quantile(errors, q_adj) * scale
        quantiles[_qname(q)] = float(point_forecast + q_val)

    return quantiles


def adaptive_conformal_intervals(
    y: np.ndarray,
    point_forecast: float,
    horizon: int,
    window: int = 104
) -> Dict[str, float]:
    """
    Adaptive Conformal Inference (ACI) with sliding window.

    Uses a rolling window of recent forecast errors, adapting to
    changing volatility regimes.

    Args:
        y: Historical time series
        point_forecast: Point forecast for target
        horizon: Forecast horizon
        window: Size of sliding window for error computation

    Returns:
        Dict with quantile values
    """
    n = len(y)
    lookback = min(window, n - horizon - 1)

    if lookback < 10:
        lookback = n - horizon - 1

    # Compute forecast errors over sliding window
    forecast_errors = []
    start_idx = max(0, n - horizon - lookback)

    for t in range(start_idx, n - horizon):
        forecast = y[t]
        actual = y[t + horizon]
        forecast_errors.append(actual - forecast)

    if len(forecast_errors) < 3:
        # Fallback
        forecast_errors = np.diff(y[-20:]).tolist()

    errors = np.array(forecast_errors)
    n_cal = len(errors)

    # Horizon scaling
    scale = 1.0 + 0.1 * horizon

    # Generate quantiles with conformal correction
    quantiles = {}
    for q in QGRID_23:
        q_adj = min(0.999, max(0.001, (q * (n_cal + 1)) / n_cal))
        q_val = np.quantile(errors, q_adj) * scale
        quantiles[_qname(q)] = float(point_forecast + q_val)

    return quantiles


def conformal_full_quantiles(
    y: np.ndarray,
    point_forecast: float,
    horizon: int,
    lookback: int = 104
) -> Dict[str, float]:
    """
    Conformal intervals using TRUE forecast residuals.

    This is the main comparison method for the paper. It uses the same:
    - Lookback window (104)
    - Horizon scaling (1 + 0.1*h)
    - Point forecast (persistence)

    But critically different:
    - Empirical: residuals = y - mean(y)  [level deviations]
    - Conformal: residuals = y[t+h] - y[t]  [actual forecast errors]

    The conformal approach models the actual distribution of forecast
    errors, while the empirical approach models level variability.

    Args:
        y: Historical time series
        point_forecast: Point forecast (persistence)
        horizon: Forecast horizon
        lookback: Window size for error computation

    Returns:
        Dict with quantile values
    """
    n = len(y)
    effective_lookback = min(lookback, n - horizon - 1)

    if effective_lookback < 10:
        effective_lookback = max(10, n - horizon - 1)

    # Compute TRUE forecast errors: what would h-step persistence have done?
    forecast_errors = []
    start_idx = max(0, n - horizon - effective_lookback)

    for t in range(start_idx, n - horizon):
        # Persistence forecast at time t for time t+horizon
        persistence_forecast = y[t]
        actual_value = y[t + horizon]
        error = actual_value - persistence_forecast
        forecast_errors.append(error)

    if len(forecast_errors) < 5:
        # Very short series fallback: use step differences
        forecast_errors = []
        for t in range(max(0, n - 30), n - 1):
            forecast_errors.append(y[t + 1] - y[t])

    errors = np.array(forecast_errors)
    n_cal = len(errors)

    # Horizon scaling (same as empirical)
    scale = 1.0 + 0.1 * horizon

    # Generate quantiles with conformal finite-sample correction
    quantiles = {}
    for q in QGRID_23:
        # Standard conformal correction
        if q <= 0.5:
            q_adj = max(0.001, (q * (n_cal + 1) - 1) / n_cal)
        else:
            q_adj = min(0.999, (q * (n_cal + 1)) / n_cal)

        q_val = np.quantile(errors, q_adj) * scale
        quantiles[_qname(q)] = float(point_forecast + q_val)

    return quantiles


# ============================================================================
# LEVEL-BASED VERSION (for direct comparison showing the difference)
# ============================================================================

def level_based_quantiles(
    y: np.ndarray,
    point_forecast: float,
    horizon: int,
    lookback: int = 104
) -> Dict[str, float]:
    """
    Level-based intervals (same as empirical method).

    This is equivalent to the empirical method for direct comparison.
    Included here to show the difference in residual definitions.

    Residuals: y - mean(y)  [how far from average level?]
    NOT: actual - forecast  [how wrong was the forecast?]
    """
    lookback = min(lookback, len(y))
    recent = y[-lookback:]
    baseline = np.mean(recent)
    residuals = recent - baseline

    n_cal = len(residuals)
    scale = 1.0 + 0.1 * horizon

    quantiles = {}
    for q in QGRID_23:
        # Conformal correction
        if q <= 0.5:
            q_adj = max(0.001, (q * (n_cal + 1) - 1) / n_cal)
        else:
            q_adj = min(0.999, (q * (n_cal + 1)) / n_cal)

        q_val = np.quantile(residuals, q_adj) * scale
        quantiles[_qname(q)] = float(point_forecast + q_val)

    return quantiles


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CONFORMAL PREDICTION BASELINE - TRUE VS LEVEL-BASED")
    print("=" * 70)

    np.random.seed(42)

    # Create test series with known properties
    # Random walk with drift
    n = 150
    y = np.cumsum(np.random.randn(n) * 2 + 0.1) + 100

    point = y[-1]

    print(f"\nSeries: {n} observations, last value = {point:.2f}")
    print(f"Series mean: {np.mean(y):.2f}, std: {np.std(y):.2f}")

    print("\n" + "=" * 70)
    print("COMPARISON: TRUE CONFORMAL vs LEVEL-BASED")
    print("=" * 70)

    print("\n90% Prediction Intervals by horizon:")
    print("-" * 70)
    print(f"{'Horizon':>8} {'True Conformal':>25} {'Level-Based':>25}")
    print(f"{'':>8} {'[q0.05, q0.95]':>25} {'[q0.05, q0.95]':>25}")
    print("-" * 70)

    for h in [1, 2, 4, 8]:
        # True conformal (forecast residuals)
        conf = conformal_full_quantiles(y, point, h)

        # Level-based (same as empirical)
        level = level_based_quantiles(y, point, h)

        conf_interval = f"[{conf['q0_05']:.1f}, {conf['q0_95']:.1f}]"
        level_interval = f"[{level['q0_05']:.1f}, {level['q0_95']:.1f}]"

        print(f"{h:>8} {conf_interval:>25} {level_interval:>25}")

    print("-" * 70)

    print("\n" + "=" * 70)
    print("KEY DIFFERENCE IN RESIDUAL DEFINITIONS:")
    print("=" * 70)
    print("""
    Level-based (Empirical Method):
        residuals = y[-lookback:] - mean(y[-lookback:])
        Captures: How far is each observation from the average level?

    True Conformal:
        residuals = y[t+h] - y[t]  (for each t in calibration window)
        Captures: How wrong would an h-step persistence forecast have been?

    For random walks: Level-based residuals have variance proportional to
    lookback window, while forecast errors have variance proportional to
    horizon. This creates different interval behaviors.
    """)
    print("=" * 70)
