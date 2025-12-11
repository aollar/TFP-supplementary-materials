"""
Strong SimpleTheta Baseline - Classical Theta Method (Assimakopoulos/Hyndman)
==============================================================================

A credible, literature-consistent Simple Theta baseline for cross-domain comparison.

This implementation follows:
- Assimakopoulos & Nikolopoulos (2000) - Original Theta method
- Hyndman & Billah (2003) - Improved understanding of Theta

Key properties:
- Two theta lines: θ=0 (linear trend) and θ=2 (doubled curvature)
- Classical combination: average of the two theta lines
- STL decomposition for seasonal adjustment (when period > 1)
- SES (Simple Exponential Smoothing) for level estimation
- Uses shared IntervalLawV2 for intervals (same as TFP)

This is NOT an intentionally weak baseline - it's a strong, robust forecaster.

Reference: "The theta model: a decomposition approach to forecasting"
          International Journal of Forecasting, 2000
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize_scalar

# Import shared interval law
from cross_domain_eval.interval_law_v2 import (
    interval_law_v2,
    IntervalConfig,
    QGRID_23,
    _qname
)


@dataclass
class ThetaResult:
    """Result from Theta forecast."""
    point: float
    horizon: int
    theta0_forecast: float  # Linear trend extrapolation
    theta2_forecast: float  # SES with drift
    alpha: float            # Optimal SES alpha
    drift: float            # Estimated drift per period
    seasonal_component: float  # Seasonal adjustment (0 if no seasonality)


def _optimal_ses_alpha(y: np.ndarray, max_alpha: float = 0.99) -> Tuple[float, float]:
    """
    Find optimal SES alpha by minimizing one-step-ahead MSE.

    Args:
        y: Time series values
        max_alpha: Maximum alpha value

    Returns:
        Tuple of (optimal_alpha, final_level)
    """
    n = len(y)
    if n < 3:
        return 0.3, float(y[-1])

    def ses_mse(alpha):
        """Compute MSE for given alpha."""
        level = y[0]
        mse = 0.0
        for i in range(1, n):
            error = y[i] - level
            mse += error ** 2
            level = alpha * y[i] + (1 - alpha) * level
        return mse / (n - 1)

    # Optimize alpha
    result = minimize_scalar(ses_mse, bounds=(0.01, max_alpha), method='bounded')
    optimal_alpha = result.x

    # Compute final level with optimal alpha
    level = y[0]
    for i in range(1, n):
        level = optimal_alpha * y[i] + (1 - optimal_alpha) * level

    return float(optimal_alpha), float(level)


def _stl_decompose(y: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple STL-like decomposition into trend, seasonal, and residual.

    Uses moving average for trend and cycle-subseries for seasonal.

    Args:
        y: Time series values
        period: Seasonal period

    Returns:
        Tuple of (trend, seasonal, residual) arrays
    """
    n = len(y)

    if period <= 1 or n < 2 * period:
        # No seasonality - return level as trend
        trend = np.full(n, np.mean(y))
        seasonal = np.zeros(n)
        residual = y - trend
        return trend, seasonal, residual

    # Estimate trend using centered moving average
    half_window = period // 2
    trend = np.zeros(n)

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        trend[i] = np.mean(y[start:end])

    # Detrended series
    detrended = y - trend

    # Estimate seasonal pattern using cycle-subseries means
    seasonal_pattern = np.zeros(period)
    for s in range(period):
        indices = np.arange(s, n, period)
        if len(indices) > 0:
            seasonal_pattern[s] = np.mean(detrended[indices])

    # Center the seasonal pattern
    seasonal_pattern -= np.mean(seasonal_pattern)

    # Tile seasonal pattern to full length
    seasonal = np.tile(seasonal_pattern, (n // period) + 1)[:n]

    # Residual
    residual = y - trend - seasonal

    return trend, seasonal, residual


def _estimate_drift(y: np.ndarray, window: int = None) -> float:
    """
    Estimate drift (trend slope) from the series.

    Uses robust linear regression on recent data.

    Args:
        y: Time series values
        window: Window for drift estimation (None = use all)

    Returns:
        Drift per period
    """
    n = len(y)
    if n < 2:
        return 0.0

    if window is not None:
        y = y[-window:]
        n = len(y)

    # Simple linear regression
    x = np.arange(n)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if abs(denominator) < 1e-10:
        return 0.0

    return float(numerator / denominator)


class StrongSimpleTheta:
    """
    Strong Simple Theta Forecaster.

    Implements classical Theta method with:
    - Two theta lines (θ=0 and θ=2) combined
    - STL seasonal decomposition
    - Optimal SES alpha selection
    - Shared IntervalLawV2 for probabilistic forecasts
    """

    def __init__(
        self,
        period: int = 1,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None
    ):
        """
        Initialize Strong SimpleTheta.

        Args:
            period: Seasonal period (1 = no seasonality, 12 = monthly, 52 = weekly)
            lower_bound: Lower bound for forecasts (e.g., 0 for counts)
            upper_bound: Upper bound for forecasts (e.g., 100 for percentages)
        """
        self.period = period
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Will be set after fit()
        self.y_hist = None
        self.trend = None
        self.seasonal = None
        self.residual = None
        self.alpha = None
        self.ses_level = None
        self.global_slope = None  # Global slope b from regression (used for drift)
        self.seasonal_pattern = None

    def fit(self, y: np.ndarray) -> 'StrongSimpleTheta':
        """
        Fit the Theta model to historical data.

        Args:
            y: Historical time series (numpy array)

        Returns:
            self
        """
        self.y_hist = np.asarray(y).flatten()
        n = len(self.y_hist)

        if n < 3:
            raise ValueError("Need at least 3 observations")

        # Step 1: STL decomposition (if seasonal)
        if self.period > 1 and n >= 2 * self.period:
            self.trend, self.seasonal, self.residual = _stl_decompose(
                self.y_hist, self.period
            )
            # Store seasonal pattern for forecasting
            self.seasonal_pattern = self.seasonal[-self.period:]
        else:
            self.trend = self.y_hist.copy()
            self.seasonal = np.zeros(n)
            self.residual = np.zeros(n)
            self.seasonal_pattern = np.zeros(max(1, self.period))

        # Step 2: Work with seasonally adjusted series
        y_sa = self.y_hist - self.seasonal

        # Step 3: Optimal SES on seasonally adjusted series
        self.alpha, self.ses_level = _optimal_ses_alpha(y_sa)

        # Step 4: Compute global slope from linear regression on full history
        # This slope b is used for both theta-0 and theta-2 (drift = b * h / 2)
        t = np.arange(n)
        try:
            coeffs = np.polyfit(t, y_sa, deg=1)
            self.global_slope = float(coeffs[0])  # slope b
        except:
            self.global_slope = 0.0

        return self

    def forecast_point(self, horizon: int) -> ThetaResult:
        """
        Generate point forecast for a single horizon.

        Combines two theta lines:
        - θ=0: Linear trend extrapolation
        - θ=2: SES level + drift (emphasizes recent level)

        Args:
            horizon: Forecast horizon (1, 2, 3, ...)

        Returns:
            ThetaResult with point forecast and components
        """
        if self.y_hist is None:
            raise ValueError("Must call fit() before forecast()")

        n = len(self.y_hist)
        y_sa = self.y_hist - self.seasonal

        # Theta line 0: Linear trend extrapolation
        # Use global slope b from fit(), forecast as a + b * (n + h)
        # Note: polyfit returns [slope, intercept], and we use (n-1+horizon) for 0-indexed
        x = np.arange(n)
        coeffs = np.polyfit(x, y_sa, deg=1)
        slope, intercept = coeffs[0], coeffs[1]
        theta0_forecast = intercept + slope * (n - 1 + horizon)

        # Theta line 2: SES level with drift
        # drift(h) = b * h / 2 where b is the global slope from regression
        drift_h = self.global_slope * horizon / 2
        theta2_forecast = self.ses_level + drift_h

        # Classical Theta combination: simple average of θ=0 and θ=2
        point_sa = 0.5 * theta0_forecast + 0.5 * theta2_forecast

        # Add seasonal component
        if self.period > 1 and len(self.seasonal_pattern) == self.period:
            seasonal_idx = (n + horizon - 1) % self.period
            seasonal_component = self.seasonal_pattern[seasonal_idx]
        else:
            seasonal_component = 0.0

        point = point_sa + seasonal_component

        # Apply bounds
        if self.lower_bound is not None:
            point = max(point, self.lower_bound)
        if self.upper_bound is not None:
            point = min(point, self.upper_bound)

        return ThetaResult(
            point=float(point),
            horizon=horizon,
            theta0_forecast=float(theta0_forecast + seasonal_component),
            theta2_forecast=float(theta2_forecast + seasonal_component),
            alpha=self.alpha,
            drift=drift_h,  # drift(h) = b * h / 2 for this horizon
            seasonal_component=seasonal_component
        )

    def forecast(
        self,
        horizon: int,
        include_intervals: bool = True
    ) -> Dict:
        """
        Generate probabilistic forecast with intervals.

        Uses shared IntervalLawV2 for interval generation.

        Args:
            horizon: Forecast horizon
            include_intervals: If True, include quantile forecasts

        Returns:
            Dict with 'point', 'quantiles', 'theta_result'
        """
        theta_result = self.forecast_point(horizon)

        result = {
            'point': theta_result.point,
            'horizon': horizon,
            'theta_result': theta_result
        }

        if include_intervals:
            config = IntervalConfig(
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound
            )
            quantiles = interval_law_v2(
                self.y_hist,
                theta_result.point,
                horizon,
                config
            )
            result['quantiles'] = quantiles

        return result

    def forecast_multi_horizon(
        self,
        horizons: List[int],
        include_intervals: bool = True
    ) -> List[Dict]:
        """
        Generate forecasts for multiple horizons.

        Args:
            horizons: List of horizons (e.g., [1, 2, 3, 4])
            include_intervals: If True, include quantile forecasts

        Returns:
            List of forecast dicts
        """
        return [self.forecast(h, include_intervals) for h in horizons]


def simple_theta_forecast(
    y: np.ndarray,
    horizons: List[int],
    period: int = 1,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None
) -> List[Dict]:
    """
    Convenience function for Strong SimpleTheta forecasting.

    Args:
        y: Historical time series
        horizons: List of forecast horizons
        period: Seasonal period
        lower_bound: Lower bound for forecasts
        upper_bound: Upper bound for forecasts

    Returns:
        List of forecast dicts with 'point' and 'quantiles'
    """
    model = StrongSimpleTheta(
        period=period,
        lower_bound=lower_bound,
        upper_bound=upper_bound
    )
    model.fit(y)
    return model.forecast_multi_horizon(horizons)


# =============================================================================
# TESTING
# =============================================================================

def main():
    """Test Strong SimpleTheta with sample data."""
    print("=" * 80)
    print("STRONG SIMPLE THETA BASELINE")
    print("=" * 80)
    print()

    np.random.seed(42)

    # Test 1: Weekly seasonal data
    print("Test 1: Weekly seasonal data (period=52)")
    t = np.arange(156)  # 3 years
    seasonal = 20 * np.sin(2 * np.pi * t / 52)
    trend = 100 + 0.5 * t
    noise = np.random.normal(0, 5, len(t))
    y_weekly = trend + seasonal + noise

    model = StrongSimpleTheta(period=52, lower_bound=0)
    model.fit(y_weekly)

    print(f"  History length: {len(y_weekly)}")
    print(f"  Optimal alpha: {model.alpha:.3f}")
    print(f"  Global slope (b): {model.global_slope:.3f}")
    print()
    print("  Forecasts:")

    for h in [1, 2, 3, 4]:
        result = model.forecast(h)
        print(f"    H{h}: point={result['point']:.2f}, "
              f"90% CI=[{result['quantiles']['q0_05']:.2f}, "
              f"{result['quantiles']['q0_95']:.2f}]")

    print()

    # Test 2: Non-seasonal data
    print("Test 2: Non-seasonal data (period=1)")
    y_nonseasonal = 50 + 0.3 * np.arange(100) + np.random.normal(0, 3, 100)

    model2 = StrongSimpleTheta(period=1)
    model2.fit(y_nonseasonal)

    print(f"  History length: {len(y_nonseasonal)}")
    print(f"  Optimal alpha: {model2.alpha:.3f}")
    print(f"  Global slope (b): {model2.global_slope:.3f}")
    print()
    print("  Forecasts:")

    for h in [1, 2, 3, 4]:
        result = model2.forecast(h)
        print(f"    H{h}: point={result['point']:.2f}, "
              f"90% CI=[{result['quantiles']['q0_05']:.2f}, "
              f"{result['quantiles']['q0_95']:.2f}]")

    print()
    print("Strong SimpleTheta baseline implementation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
