#!/usr/bin/env python3
"""
Generate Figure 1 for BASS_PAPER_DRAFT.md
Bar chart comparing MAE across models on technology adoption forecasting.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from Table 1 in paper (MAE by Model - All Windows)
models = ['TFP v2.2', 'Naive', 'SimpleTheta', 'Logistic*', 'Gompertz*', 'Bass*']
mae_values = [2.81, 3.13, 4.35, 5.43, 5.66, 6.36]
n_windows = [840, 840, 840, 513, 513, 513]

# Colors: TFP in distinct color, non-parametric in medium gray, parametric in dark gray
colors = ['#2E86AB', '#888888', '#888888', '#555555', '#555555', '#555555']

# Create figure with extra space at bottom for note
fig, ax = plt.subplots(figsize=(10, 7))

# Horizontal bar chart
y_pos = np.arange(len(models))
bars = ax.barh(y_pos, mae_values, color=colors, edgecolor='black', linewidth=0.5, height=0.7)

# Add value labels on bars (outside the bars for clarity)
for i, (bar, mae, n) in enumerate(zip(bars, mae_values, n_windows)):
    width = bar.get_width()
    label = f'{mae:.2f}%'
    ax.text(width + 0.15, bar.get_y() + bar.get_height()/2,
            label, va='center', fontsize=12, fontweight='bold')

# Customize axes
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=13)
ax.invert_yaxis()  # Best performer at top
ax.set_xlabel('Mean Absolute Error (percentage points)', fontsize=12)
ax.set_title('Figure 1: MAE Comparison Across Forecasting Models\n21 US Technology Adoption Curves',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, 8.5)

# Add vertical line at TFP's MAE for reference
ax.axvline(x=2.81, color='#2E86AB', linestyle='--', alpha=0.4, linewidth=2)

# Add improvement annotation - positioned clearly between bars
ax.annotate('',
            xy=(2.81, 5.15), xytext=(6.36, 5.15),
            arrowprops=dict(arrowstyle='<->', color='#2E86AB', lw=2))
ax.text(4.6, 5.55, '34% improvement', fontsize=11, ha='center',
        color='#2E86AB', fontweight='bold')

# Add grid for readability
ax.xaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend/note below the chart
fig.text(0.5, 0.02,
         '* Parametric models (Bass, Gompertz, Logistic) evaluated on 513 windows where fitting converged.\n'
         '  Non-parametric methods (TFP, Naive, SimpleTheta) evaluated on all 840 windows.',
         ha='center', fontsize=10, style='italic', color='#444444')

# Adjust layout to make room for the note
plt.subplots_adjust(bottom=0.15)

plt.savefig('/home/user/TFP-core/Bass-Research/figures/figure1_mae_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/user/TFP-core/Bass-Research/figures/figure1_mae_comparison.pdf',
            bbox_inches='tight', facecolor='white')
print("Figure saved to figures/figure1_mae_comparison.png and .pdf")
