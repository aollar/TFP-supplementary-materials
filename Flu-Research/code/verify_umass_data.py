"""
UMass Data Verification Script

Compares UMass.fix.csv against raw UMass files in Flu-Update/Umass/
to verify data integrity.

Expected filters for UMass.fix.csv:
- horizons: 1, 2, 3 only (no 0 or -1)
- output_type: 'quantile' only
- target: 'wk inc flu hosp'
"""

import pandas as pd
import os
from glob import glob

def main():
    print("=" * 80)
    print("UMASS DATA VERIFICATION")
    print("=" * 80)
    print()

    # Load UMass.fix.csv
    fix_path = '/home/user/TFP-core/UMass.fix.csv'
    print(f"Loading {fix_path}...")
    fix_df = pd.read_csv(fix_path)
    fix_df['forecast_date'] = pd.to_datetime(fix_df['forecast_date'])
    print(f"  Rows: {len(fix_df):,}")
    print(f"  Columns: {list(fix_df.columns)}")
    print()

    # Check fix_df filters
    print("Checking UMass.fix.csv filters:")
    print(f"  Unique horizons: {sorted(fix_df['horizon'].unique())}")
    if 'output_type' in fix_df.columns:
        print(f"  Unique output_type: {fix_df['output_type'].unique()}")
    if 'target' in fix_df.columns:
        print(f"  Unique target: {fix_df['target'].unique()}")
    print()

    # Load raw UMass files
    raw_dir = '/home/user/TFP-core/Flu-Update/Umass/'
    raw_files = sorted(glob(os.path.join(raw_dir, '*.csv')))
    print(f"Found {len(raw_files)} raw UMass files in {raw_dir}")
    print()

    # Combine all raw files
    print("Loading and combining raw files...")
    raw_dfs = []
    for f in raw_files:
        df = pd.read_csv(f)
        # Extract forecast date from filename
        fname = os.path.basename(f)
        fc_date = fname.split('-UMass')[0]
        df['forecast_date'] = pd.to_datetime(fc_date)
        raw_dfs.append(df)

    raw_df = pd.concat(raw_dfs, ignore_index=True)
    print(f"  Total raw rows: {len(raw_df):,}")
    print(f"  Columns: {list(raw_df.columns)}")
    print()

    # Check raw data characteristics
    print("Raw data characteristics:")
    print(f"  Unique horizons: {sorted(raw_df['horizon'].unique())}")
    if 'output_type' in raw_df.columns:
        print(f"  Unique output_type: {raw_df['output_type'].unique()}")
    if 'target' in raw_df.columns:
        print(f"  Unique target: {raw_df['target'].unique()}")
    print()

    # Apply expected filters to raw data
    print("Applying filters to raw data:")
    print("  - horizon in [1, 2, 3]")
    print("  - output_type == 'quantile'")
    print("  - target == 'wk inc flu hosp'")

    filtered_raw = raw_df[
        (raw_df['horizon'].isin([1, 2, 3])) &
        (raw_df['output_type'] == 'quantile') &
        (raw_df['target'] == 'wk inc flu hosp')
    ].copy()

    print(f"  Filtered raw rows: {len(filtered_raw):,}")
    print()

    # Compare row counts
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()

    print(f"UMass.fix.csv rows:    {len(fix_df):,}")
    print(f"Filtered raw rows:     {len(filtered_raw):,}")
    print(f"Difference:            {len(fix_df) - len(filtered_raw):,}")
    print()

    # Compare by forecast date
    print("Rows by forecast date:")
    fix_by_date = fix_df.groupby('forecast_date').size()
    raw_by_date = filtered_raw.groupby('forecast_date').size()

    all_dates = sorted(set(fix_by_date.index) | set(raw_by_date.index))

    print(f"{'Date':<15} {'UMass.fix':>12} {'Raw Filtered':>15} {'Match':>8}")
    print("-" * 55)

    mismatches = []
    for d in all_dates:
        fix_n = fix_by_date.get(d, 0)
        raw_n = raw_by_date.get(d, 0)
        match = "✓" if fix_n == raw_n else "✗"
        if fix_n != raw_n:
            mismatches.append((d, fix_n, raw_n))
        print(f"{str(d.date()):<15} {fix_n:>12,} {raw_n:>15,} {match:>8}")

    print()

    # Check for missing dates
    fix_dates = set(fix_by_date.index)
    raw_dates = set(raw_by_date.index)

    if fix_dates - raw_dates:
        print(f"Dates in UMass.fix but NOT in raw: {sorted(fix_dates - raw_dates)}")
    if raw_dates - fix_dates:
        print(f"Dates in raw but NOT in UMass.fix: {sorted(raw_dates - fix_dates)}")

    # Check value column
    print()
    print("Checking value column naming:")
    print(f"  UMass.fix.csv value column: {'umass_pred_value' if 'umass_pred_value' in fix_df.columns else 'value'}")
    print(f"  Raw data value column: {'value' if 'value' in filtered_raw.columns else 'unknown'}")

    # Try to match specific rows
    print()
    print("=" * 80)
    print("SPOT CHECK: First forecast date")
    print("=" * 80)

    first_date = min(all_dates)
    fix_first = fix_df[fix_df['forecast_date'] == first_date]
    raw_first = filtered_raw[filtered_raw['forecast_date'] == first_date]

    print(f"Date: {first_date.date()}")
    print(f"UMass.fix rows: {len(fix_first)}")
    print(f"Raw filtered rows: {len(raw_first)}")
    print()

    # Compare unique locations
    fix_locs = set(fix_first['location'].unique())
    raw_locs = set(raw_first['location'].unique())
    print(f"Locations in both: {len(fix_locs & raw_locs)}")
    print(f"Locations only in fix: {fix_locs - raw_locs}")
    print(f"Locations only in raw: {raw_locs - fix_locs}")
    print()

    # Check a specific location/horizon combo
    loc = 'US'
    h = 1
    fix_sample = fix_first[(fix_first['location'] == loc) & (fix_first['horizon'] == h)]
    raw_sample = raw_first[(raw_first['location'] == loc) & (raw_first['horizon'] == h)]

    print(f"Sample: location={loc}, horizon={h}")
    print(f"  UMass.fix rows: {len(fix_sample)}")
    print(f"  Raw filtered rows: {len(raw_sample)}")

    if len(fix_sample) > 0 and len(raw_sample) > 0:
        # Get the value column
        fix_val_col = 'umass_pred_value' if 'umass_pred_value' in fix_sample.columns else 'value'
        raw_val_col = 'value'

        fix_vals = fix_sample.sort_values('output_type_id')[fix_val_col].values
        raw_vals = raw_sample.sort_values('output_type_id')[raw_val_col].values

        if len(fix_vals) == len(raw_vals):
            max_diff = abs(fix_vals - raw_vals).max()
            print(f"  Max value difference: {max_diff}")
            if max_diff < 0.001:
                print("  ✓ Values match!")
            else:
                print("  ✗ Values differ!")
        else:
            print(f"  Different number of quantiles: {len(fix_vals)} vs {len(raw_vals)}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if len(mismatches) == 0 and len(fix_df) == len(filtered_raw):
        print("✓ UMass.fix.csv matches raw data after filtering!")
        print("  - All forecast dates have matching row counts")
        print("  - Filters applied: horizon in [1,2,3], output_type='quantile', target='wk inc flu hosp'")
    else:
        print("✗ Discrepancies found:")
        print(f"  - Row count difference: {len(fix_df) - len(filtered_raw)}")
        print(f"  - Dates with mismatches: {len(mismatches)}")
        for d, fix_n, raw_n in mismatches[:5]:
            print(f"    {d.date()}: fix={fix_n}, raw={raw_n}")


if __name__ == "__main__":
    main()
