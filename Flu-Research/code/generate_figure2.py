#!/usr/bin/env python3
"""
Generate Figure 2: H1 Forecasts for California, Dec 2024 - Jan 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Read the results data
df = pd.read_csv('results/FLU_STATES_PUBLICATION_RESULTS_intervalLawAll.csv')

# Filter for California (location 06), horizon 1
ca_h1 = df[(df['location'] == '06') & (df['horizon'] == 1)].copy()

# Convert dates
ca_h1['target_end_date'] = pd.to_datetime(ca_h1['target_end_date'])

# Filter for Dec 2024 - Jan 2025 peak period
mask = (ca_h1['target_end_date'] >= '2024-12-01') & (ca_h1['target_end_date'] <= '2025-01-31')
ca_peak = ca_h1[mask].sort_values('target_end_date')

print(f"Found {len(ca_peak)} data points for California H1 in Dec 2024 - Jan 2025")
print(ca_peak[['target_end_date', 'actual', 'tfp_point', 'umass_point', 'cdc_point']].to_string())

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot actual values
ax.plot(ca_peak['target_end_date'], ca_peak['actual'],
        'ko-', markersize=8, linewidth=2, label='Actual', zorder=5)

# Plot forecasts
ax.plot(ca_peak['target_end_date'], ca_peak['tfp_point'],
        'b^-', markersize=7, linewidth=1.5, label='TFP v2.2', alpha=0.9)
ax.plot(ca_peak['target_end_date'], ca_peak['umass_point'],
        's-', color='#FF7F0E', markersize=7, linewidth=1.5, label='UMass-Flusion', alpha=0.9)
ax.plot(ca_peak['target_end_date'], ca_peak['cdc_point'],
        'd-', color='#2CA02C', markersize=7, linewidth=1.5, label='FluSight Ensemble', alpha=0.9)

# Formatting
ax.set_xlabel('Target Week Ending', fontsize=12)
ax.set_ylabel('Hospitalizations', fontsize=12)
ax.set_title('One-Week-Ahead Flu Forecasts: California, Dec 2024 - Jan 2025', fontsize=14, fontweight='bold')

# Format x-axis dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45, ha='right')

# Legend
ax.legend(loc='upper left', fontsize=10)

# Grid
ax.grid(True, alpha=0.3)

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('figures/ca_h1_example.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/ca_h1_example.pdf', bbox_inches='tight')
print("\nFigure saved to figures/ca_h1_example.png and .pdf")

# Print MAE comparison for this period
print(f"\n--- MAE for California H1, Dec 2024 - Jan 2025 ---")
print(f"TFP MAE:     {ca_peak['tfp_mae'].mean():.1f}")
print(f"UMass MAE:   {ca_peak['umass_mae'].mean():.1f}")
print(f"FluSight MAE: {ca_peak['cdc_mae'].mean():.1f}")
