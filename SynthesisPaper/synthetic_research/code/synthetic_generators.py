"""
Synthetic Data Generators for TFP Law-Like Generalist Benchmark
================================================================

Three families of synthetic series with known structure:
1. Trend + Noise: Piecewise linear trends with potential turning points
2. Seasonal AR(1): Strong seasonality with autoregressive level process
3. Logistic S-curve: Technology adoption-like curves in growth and saturation regimes

All generators use np.random.default_rng(seed) for reproducibility.
"""

import numpy as np
from typing import Tuple, List


def generate_trend_series(n_series: int = 500, seed: int = 1) -> Tuple[np.ndarray, List[dict]]:
    """
    Generate Trend + Noise series.

    Structure:
    - Length: 200 time steps
    - First 100 points: linear trend with random slope in [-0.5, +0.5]
    - Last 100 points: 50% chance to flip slope sign (turning point)
    - Gaussian noise with std = 5-15% of the signal range

    Args:
        n_series: Number of series to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - Array of shape (n_series, 200)
        - List of metadata dicts for each series
    """
    rng = np.random.default_rng(seed)
    T = 200

    series_data = np.zeros((n_series, T))
    metadata = []

    for i in range(n_series):
        # Initial level
        level = rng.uniform(50, 150)

        # First segment slope
        slope1 = rng.uniform(-0.5, 0.5)

        # Decide if turning point
        has_turning_point = rng.random() < 0.5
        if has_turning_point:
            slope2 = -slope1 * rng.uniform(0.5, 1.5)  # Flip and randomize magnitude
        else:
            slope2 = slope1 * rng.uniform(0.8, 1.2)  # Continue with slight change

        # Generate underlying mean
        mean = np.zeros(T)
        for t in range(100):
            mean[t] = level + slope1 * t
        level_100 = mean[99]
        for t in range(100, T):
            mean[t] = level_100 + slope2 * (t - 99)

        # Compute signal range for noise calibration
        signal_range = np.max(mean) - np.min(mean)
        if signal_range < 1.0:
            signal_range = 10.0  # Avoid division issues for flat series

        # Noise std: 5-15% of range
        noise_pct = rng.uniform(0.05, 0.15)
        noise_std = noise_pct * signal_range

        # Add noise
        noise = rng.normal(0, noise_std, T)
        series_data[i] = mean + noise

        metadata.append({
            'family': 'trend',
            'series_id': i,
            'initial_level': level,
            'slope1': slope1,
            'slope2': slope2,
            'has_turning_point': has_turning_point,
            'noise_std': noise_std,
        })

    return series_data, metadata


def generate_seasonal_series(n_series: int = 500, seed: int = 1,
                             period: int = 12) -> Tuple[np.ndarray, List[dict]]:
    """
    Generate Seasonal AR(1) series.

    Structure:
    - Length: 240 time steps
    - Strong seasonality with period 12 (monthly)
    - Level process follows AR(1) with coefficient in [0.3, 0.9]
    - Seasonal pattern amplitude and phase are randomized
    - Gaussian noise on top

    Args:
        n_series: Number of series to generate
        seed: Random seed for reproducibility
        period: Seasonal period (default 12 for monthly)

    Returns:
        Tuple of:
        - Array of shape (n_series, 240)
        - List of metadata dicts for each series
    """
    rng = np.random.default_rng(seed)
    T = 240

    series_data = np.zeros((n_series, T))
    metadata = []

    for i in range(n_series):
        # AR(1) coefficient
        ar_coef = rng.uniform(0.3, 0.9)

        # Base level and long-term mean
        base_level = rng.uniform(50, 150)

        # Seasonal pattern (sinusoidal with random phase and amplitude)
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = rng.uniform(10, 40)  # Seasonal swing

        seasonal = amplitude * np.sin(2 * np.pi * np.arange(T) / period + phase)

        # Generate AR(1) level process
        level = np.zeros(T)
        level[0] = base_level
        innovation_std = rng.uniform(2, 8)

        for t in range(1, T):
            innovation = rng.normal(0, innovation_std)
            level[t] = base_level * (1 - ar_coef) + ar_coef * level[t-1] + innovation

        # Combine level + seasonal
        mean = level + seasonal

        # Add observation noise
        signal_range = np.max(mean) - np.min(mean)
        if signal_range < 1.0:
            signal_range = 20.0
        noise_pct = rng.uniform(0.03, 0.10)
        noise_std = noise_pct * signal_range

        noise = rng.normal(0, noise_std, T)
        series_data[i] = mean + noise

        metadata.append({
            'family': 'seasonal',
            'series_id': i,
            'ar_coef': ar_coef,
            'base_level': base_level,
            'amplitude': amplitude,
            'phase': phase,
            'period': period,
            'innovation_std': innovation_std,
            'noise_std': noise_std,
        })

    return series_data, metadata


def generate_logistic_series(n_series: int = 500, regime: str = 'growth',
                             seed: int = 1) -> Tuple[np.ndarray, List[dict]]:
    """
    Generate Logistic S-curve adoption series.

    Structure:
    - Length: 160 time steps
    - Underlying mean is logistic: y = K / (1 + exp(-r * (t - t0)))
    - Parameters:
        - K (asymptote): [80, 120]
        - r (growth rate): [0.03, 0.15]
        - t0 (midpoint): [50, 110]
    - Mild local wiggles + Gaussian noise

    Regimes:
    - 'growth': Evaluation windows centered in 20-80% of K
    - 'saturated': Evaluation windows centered in 90-100% of K

    Args:
        n_series: Number of series to generate
        regime: 'growth' or 'saturated'
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - Array of shape (n_series, 160)
        - List of metadata dicts for each series
    """
    rng = np.random.default_rng(seed)
    T = 160

    series_data = np.zeros((n_series, T))
    metadata = []

    for i in range(n_series):
        # Logistic parameters
        K = rng.uniform(80, 120)
        r = rng.uniform(0.03, 0.15)

        # Midpoint selection depends on regime
        if regime == 'growth':
            # t0 should be such that evaluation windows are in 20-80% of K
            # At t=T, we want y to be in [0.2K, 0.8K]
            # y = K / (1 + exp(-r*(T-t0))) => solve for t0
            # For growth regime, t0 should be later in the series
            t0 = rng.uniform(70, 130)
        else:  # saturated
            # For saturated, t0 should be earlier so series is near K by end
            t0 = rng.uniform(30, 70)

        # Generate logistic mean
        t = np.arange(T)
        mean = K / (1 + np.exp(-r * (t - t0)))

        # Add mild local wiggles (small AR(1) process)
        wiggle_coef = rng.uniform(0.5, 0.8)
        wiggle_std = rng.uniform(1, 3)
        wiggles = np.zeros(T)
        for j in range(1, T):
            wiggles[j] = wiggle_coef * wiggles[j-1] + rng.normal(0, wiggle_std)

        # Add Gaussian noise
        signal_range = K  # For logistic, K is a reasonable scale
        noise_pct = rng.uniform(0.02, 0.08)
        noise_std = noise_pct * signal_range
        noise = rng.normal(0, noise_std, T)

        series_data[i] = mean + wiggles + noise

        # Compute what percent of K we reach at the end
        final_pct = mean[-1] / K

        metadata.append({
            'family': 'logistic',
            'series_id': i,
            'regime': regime,
            'K': K,
            'r': r,
            't0': t0,
            'wiggle_coef': wiggle_coef,
            'wiggle_std': wiggle_std,
            'noise_std': noise_std,
            'final_pct_of_K': final_pct,
        })

    return series_data, metadata


def generate_trend_logistic_mix(n_series: int = 500, w_logistic: float = 0.5,
                                 seed: int = 1) -> Tuple[np.ndarray, List[dict]]:
    """
    Generate Trend + Logistic S-curve mixture series.

    Structure:
    - Length: 200 time steps (matches trend family)
    - Base component: Linear trend with noise (simplified from trend family)
    - Logistic component: S-curve rescaled to match base magnitude
    - Mixture: series = base + w_logistic * logistic_rescaled

    When w_logistic = 0.0, produces pure trend.
    When w_logistic = 1.0, produces trend + strong S-curve.

    Args:
        n_series: Number of series to generate
        w_logistic: Weight for logistic component [0.0, 1.0]
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - Array of shape (n_series, 200)
        - List of metadata dicts for each series
    """
    rng = np.random.default_rng(seed)
    T = 200

    series_data = np.zeros((n_series, T))
    metadata = []

    for i in range(n_series):
        # === BASE COMPONENT: Linear trend ===
        level = rng.uniform(50, 150)
        slope = rng.uniform(-0.3, 0.3)  # Simpler single slope

        t = np.arange(T)
        base = level + slope * t
        base_range = np.max(base) - np.min(base)
        if base_range < 10.0:
            base_range = 10.0

        # === LOGISTIC COMPONENT ===
        # Parameters for S-curve
        K = rng.uniform(0.8, 1.2)  # Normalized asymptote
        r = rng.uniform(0.03, 0.10)  # Growth rate
        t0 = rng.uniform(80, 120)  # Midpoint in middle of series

        logistic_raw = K / (1 + np.exp(-r * (t - t0)))

        # Rescale logistic to match base magnitude
        # Scale so max swing is similar to base_range
        logistic_rescaled = logistic_raw * base_range

        # === MIXTURE ===
        mean = base + w_logistic * logistic_rescaled

        # === NOISE ===
        signal_range = np.max(mean) - np.min(mean)
        if signal_range < 1.0:
            signal_range = 10.0
        noise_pct = rng.uniform(0.05, 0.12)
        noise_std = noise_pct * signal_range
        noise = rng.normal(0, noise_std, T)

        series_data[i] = mean + noise

        metadata.append({
            'family': 'trend_logistic_mix',
            'series_id': i,
            'w_logistic': w_logistic,
            'base_level': level,
            'base_slope': slope,
            'logistic_K': K,
            'logistic_r': r,
            'logistic_t0': t0,
            'noise_std': noise_std,
        })

    return series_data, metadata


def generate_seasonal_logistic_mix(n_series: int = 500, w_logistic: float = 0.5,
                                    seed: int = 1, period: int = 12) -> Tuple[np.ndarray, List[dict]]:
    """
    Generate Seasonal + Logistic S-curve mixture series.

    Structure:
    - Length: 240 time steps (matches seasonal family)
    - Base component: Seasonal pattern with AR(1) level (from seasonal family)
    - Logistic component: S-curve rescaled to match base magnitude
    - Mixture: series = base + w_logistic * logistic_rescaled

    When w_logistic = 0.0, produces pure seasonal.
    When w_logistic = 1.0, produces seasonal + strong S-curve.

    Args:
        n_series: Number of series to generate
        w_logistic: Weight for logistic component [0.0, 1.0]
        seed: Random seed for reproducibility
        period: Seasonal period (default 12)

    Returns:
        Tuple of:
        - Array of shape (n_series, 240)
        - List of metadata dicts for each series
    """
    rng = np.random.default_rng(seed)
    T = 240

    series_data = np.zeros((n_series, T))
    metadata = []

    for i in range(n_series):
        # === BASE COMPONENT: Seasonal with AR(1) level ===
        ar_coef = rng.uniform(0.5, 0.8)
        base_level = rng.uniform(50, 150)

        # Seasonal pattern
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = rng.uniform(15, 35)
        t = np.arange(T)
        seasonal = amplitude * np.sin(2 * np.pi * t / period + phase)

        # AR(1) level process
        level = np.zeros(T)
        level[0] = base_level
        innovation_std = rng.uniform(3, 6)
        for j in range(1, T):
            innovation = rng.normal(0, innovation_std)
            level[j] = base_level * (1 - ar_coef) + ar_coef * level[j-1] + innovation

        base = level + seasonal
        base_range = np.max(base) - np.min(base)
        if base_range < 20.0:
            base_range = 20.0

        # === LOGISTIC COMPONENT ===
        K = rng.uniform(0.8, 1.2)  # Normalized asymptote
        r = rng.uniform(0.02, 0.08)  # Growth rate
        t0 = rng.uniform(100, 140)  # Midpoint

        logistic_raw = K / (1 + np.exp(-r * (t - t0)))

        # Rescale logistic to match base magnitude
        logistic_rescaled = logistic_raw * base_range

        # === MIXTURE ===
        mean = base + w_logistic * logistic_rescaled

        # === NOISE ===
        signal_range = np.max(mean) - np.min(mean)
        if signal_range < 1.0:
            signal_range = 20.0
        noise_pct = rng.uniform(0.03, 0.08)
        noise_std = noise_pct * signal_range
        noise = rng.normal(0, noise_std, T)

        series_data[i] = mean + noise

        metadata.append({
            'family': 'seasonal_logistic_mix',
            'series_id': i,
            'w_logistic': w_logistic,
            'base_level': base_level,
            'ar_coef': ar_coef,
            'amplitude': amplitude,
            'phase': phase,
            'period': period,
            'logistic_K': K,
            'logistic_r': r,
            'logistic_t0': t0,
            'noise_std': noise_std,
        })

    return series_data, metadata


def generate_hybrid_families(n_series: int = 500,
                             w_logistic_grid: List[float] = None,
                             seed: int = 1) -> dict:
    """
    Generate hybrid families for all w_logistic values.

    Args:
        n_series: Number of series per family per w_logistic
        w_logistic_grid: List of logistic weights to use
        seed: Base random seed

    Returns:
        Dict mapping (family, w_logistic) to (data, metadata) tuple
    """
    if w_logistic_grid is None:
        w_logistic_grid = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    families = {}

    for idx, w in enumerate(w_logistic_grid):
        # Trend + Logistic mix
        families[('trend_logistic_mix', w)] = generate_trend_logistic_mix(
            n_series, w_logistic=w, seed=seed + idx * 100
        )

        # Seasonal + Logistic mix
        families[('seasonal_logistic_mix', w)] = generate_seasonal_logistic_mix(
            n_series, w_logistic=w, seed=seed + 1000 + idx * 100
        )

    return families


def generate_all_families(n_series: int = 500, seed: int = 1) -> dict:
    """
    Generate all synthetic families.

    Args:
        n_series: Number of series per family (per regime for logistic)
        seed: Base random seed

    Returns:
        Dict mapping family name to (data, metadata) tuple
    """
    families = {}

    # Trend + Noise
    families['trend'] = generate_trend_series(n_series, seed=seed)

    # Seasonal AR(1)
    families['seasonal'] = generate_seasonal_series(n_series, seed=seed + 1000)

    # Logistic - Growth regime
    families['logistic_growth'] = generate_logistic_series(
        n_series, regime='growth', seed=seed + 2000
    )

    # Logistic - Saturated regime
    families['logistic_saturated'] = generate_logistic_series(
        n_series, regime='saturated', seed=seed + 3000
    )

    return families


if __name__ == "__main__":
    # Quick test
    print("Generating synthetic test data...")

    trend_data, trend_meta = generate_trend_series(n_series=5, seed=42)
    print(f"Trend: shape={trend_data.shape}")
    print(f"  Sample meta: {trend_meta[0]}")

    seasonal_data, seasonal_meta = generate_seasonal_series(n_series=5, seed=42)
    print(f"Seasonal: shape={seasonal_data.shape}")
    print(f"  Sample meta: {seasonal_meta[0]}")

    logistic_g, logistic_g_meta = generate_logistic_series(n_series=5, regime='growth', seed=42)
    print(f"Logistic (growth): shape={logistic_g.shape}")
    print(f"  Sample meta: {logistic_g_meta[0]}")

    logistic_s, logistic_s_meta = generate_logistic_series(n_series=5, regime='saturated', seed=42)
    print(f"Logistic (saturated): shape={logistic_s.shape}")
    print(f"  Sample meta: {logistic_s_meta[0]}")

    print("\nAll generators working correctly!")
