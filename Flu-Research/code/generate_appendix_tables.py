#!/usr/bin/env python3
"""
Generate Appendix A tables: Detailed Per-Horizon Results by Location
"""

import pandas as pd
import numpy as np

# FIPS code to state name mapping
FIPS_TO_STATE = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas',
    '06': 'California', '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware',
    '11': 'District of Columbia', '12': 'Florida', '13': 'Georgia', '15': 'Hawaii',
    '16': 'Idaho', '17': 'Illinois', '18': 'Indiana', '19': 'Iowa',
    '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine',
    '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
    '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska',
    '32': 'Nevada', '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico',
    '36': 'New York', '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio',
    '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island',
    '45': 'South Carolina', '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas',
    '49': 'Utah', '50': 'Vermont', '51': 'Virginia', '53': 'Washington',
    '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming',
    '72': 'Puerto Rico', 'US': 'US National'
}

# Read data
df = pd.read_csv('results/FLU_STATES_PUBLICATION_RESULTS_intervalLawAll.csv')
df['location'] = df['location'].astype(str).str.zfill(2)

# Filter for 2024-2025 season
df_2425 = df[df['season'] == '2024-2025'].copy()

# Group by location and horizon
def compute_stats(group):
    return pd.Series({
        'tfp_mae': group['tfp_mae'].mean(),
        'umass_mae': group['umass_mae'].mean(),
        'cdc_mae': group['cdc_mae'].mean(),
        'n_windows': len(group)
    })

# Generate tables by horizon
print("=" * 80)
print("APPENDIX A: DETAILED PER-HORIZON RESULTS")
print("=" * 80)

for horizon in [1, 2, 3]:
    df_h = df_2425[df_2425['horizon'] == horizon]

    stats = df_h.groupby('location').apply(compute_stats).reset_index()
    stats['state'] = stats['location'].map(FIPS_TO_STATE)
    stats['tfp_vs_umass'] = ((stats['tfp_mae'] - stats['umass_mae']) / stats['umass_mae'] * 100)
    stats['tfp_vs_cdc'] = ((stats['tfp_mae'] - stats['cdc_mae']) / stats['cdc_mae'] * 100)
    stats['winner'] = stats.apply(
        lambda r: 'TFP' if r['tfp_mae'] < min(r['umass_mae'], r['cdc_mae']) else
                  ('UMass' if r['umass_mae'] < r['cdc_mae'] else 'FluSight'), axis=1
    )

    # Sort by state name
    stats = stats.sort_values('state')

    print(f"\n### A.{horizon} Horizon {horizon} (H{horizon}) Results by Location\n")
    print(f"**Table A{horizon}: MAE by Location at Horizon {horizon} (2024-2025 Season)**\n")
    print("| Location | TFP MAE | UMass MAE | FluSight MAE | TFP vs UMass | TFP vs FluSight | Winner |")
    print("|----------|---------|-----------|--------------|--------------|-----------------|--------|")

    for _, row in stats.iterrows():
        if pd.isna(row['umass_mae']):
            umass_str = "N/A"
            vs_umass = "N/A"
        else:
            umass_str = f"{row['umass_mae']:.1f}"
            vs_umass = f"{row['tfp_vs_umass']:+.1f}%"

        print(f"| {row['state']} | {row['tfp_mae']:.1f} | {umass_str} | {row['cdc_mae']:.1f} | {vs_umass} | {row['tfp_vs_cdc']:+.1f}% | {row['winner']} |")

    # Summary stats
    tfp_wins = (stats['winner'] == 'TFP').sum()
    total = len(stats)
    avg_tfp = stats['tfp_mae'].mean()
    avg_umass = stats['umass_mae'].mean()
    avg_cdc = stats['cdc_mae'].mean()

    print(f"\n**H{horizon} Summary:** TFP wins {tfp_wins}/{total} locations ({100*tfp_wins/total:.1f}%). ")
    print(f"Average MAE: TFP={avg_tfp:.1f}, UMass={avg_umass:.1f}, FluSight={avg_cdc:.1f}")

# Generate combined table
print("\n" + "=" * 80)
print("### A.4 Combined Summary by Location (All Horizons Pooled)\n")
print("**Table A4: Pooled MAE by Location (2024-2025 Season)**\n")

stats_all = df_2425.groupby('location').apply(compute_stats).reset_index()
stats_all['state'] = stats_all['location'].map(FIPS_TO_STATE)
stats_all['tfp_vs_umass'] = ((stats_all['tfp_mae'] - stats_all['umass_mae']) / stats_all['umass_mae'] * 100)
stats_all['tfp_vs_cdc'] = ((stats_all['tfp_mae'] - stats_all['cdc_mae']) / stats_all['cdc_mae'] * 100)
stats_all['winner'] = stats_all.apply(
    lambda r: 'TFP' if r['tfp_mae'] < min(r['umass_mae'], r['cdc_mae']) else
              ('UMass' if r['umass_mae'] < r['cdc_mae'] else 'FluSight'), axis=1
)
stats_all = stats_all.sort_values('state')

print("| Location | TFP MAE | UMass MAE | FluSight MAE | TFP vs UMass | TFP vs FluSight | Winner |")
print("|----------|---------|-----------|--------------|--------------|-----------------|--------|")

for _, row in stats_all.iterrows():
    if pd.isna(row['umass_mae']):
        umass_str = "N/A"
        vs_umass = "N/A"
    else:
        umass_str = f"{row['umass_mae']:.1f}"
        vs_umass = f"{row['tfp_vs_umass']:+.1f}%"

    print(f"| {row['state']} | {row['tfp_mae']:.1f} | {umass_str} | {row['cdc_mae']:.1f} | {vs_umass} | {row['tfp_vs_cdc']:+.1f}% | {row['winner']} |")

tfp_wins = (stats_all['winner'] == 'TFP').sum()
total = len(stats_all)
print(f"\n**Pooled Summary:** TFP wins {tfp_wins}/{total} locations ({100*tfp_wins/total:.1f}%)")
