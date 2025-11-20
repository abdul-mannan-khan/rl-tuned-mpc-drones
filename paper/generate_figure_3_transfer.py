"""
Generate Figure 3: Sequential Transfer Learning Flow
Professional publication-quality figure for AAAI paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Set up publication-quality parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts

# Create figure
fig, ax = plt.subplots(figsize=(8, 11))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15)
ax.axis('off')

# Define gradient blue colors for progression
colors = [
    ('#E3F2FD', '#1976D2'),  # Lightest blue
    ('#BBDEFB', '#1565C0'),  # Light blue
    ('#90CAF9', '#0D47A1'),  # Medium blue
    ('#64B5F6', '#0D47A1'),  # Darker blue
]

# Box parameters
box_width = 7
box_height = 2.2
x_center = 5

# Platform data
platforms = [
    {
        'name': 'Crazyflie 2.X',
        'mass': '0.027 kg',
        'mass_ratio': 'Baseline',
        'approach': 'Train from scratch',
        'steps': '20,000',
        'time': '200 min',
        'rmse': '1.34 m',
        'phase': 'PHASE 1: Base Training'
    },
    {
        'name': 'Racing Drone',
        'mass': '0.800 kg',
        'mass_ratio': '29.6× heavier',
        'approach': 'Load θ₁ → Fine-tune',
        'steps': '5,000 (75% ↓)',
        'time': '52 min',
        'rmse': '1.33 m',
        'phase': 'PHASE 2: Transfer Learning'
    },
    {
        'name': 'Generic Quadrotor',
        'mass': '2.500 kg',
        'mass_ratio': '92.6× heavier',
        'approach': 'Load θ₂ → Fine-tune',
        'steps': '5,000 (75% ↓)',
        'time': '52 min',
        'rmse': '1.34 m',
        'phase': 'PHASE 3: Transfer Learning'
    },
    {
        'name': 'Heavy-Lift Hexacopter',
        'mass': '5.500 kg',
        'mass_ratio': '203.7× heavier',
        'approach': 'Load θ₃ → Fine-tune',
        'steps': '5,000 (75% ↓)',
        'time': '59 min',
        'rmse': '1.34 m',
        'phase': 'PHASE 4: Transfer Learning'
    }
]

# Y positions for boxes (from top to bottom)
y_positions = [12, 9, 6, 3]

# Function to create platform box
def create_platform_box(ax, x, y, width, height, facecolor, edgecolor, platform_data, show_phase=True):
    """Create a platform box with all information"""
    # Main box
    box = FancyBboxPatch((x - width/2, y), width, height,
                          boxstyle="round,pad=0.08",
                          facecolor=facecolor,
                          edgecolor=edgecolor,
                          linewidth=2.5,
                          zorder=1)
    ax.add_patch(box)

    # Phase label (smaller, above box)
    if show_phase:
        ax.text(x, y + height + 0.15, platform_data['phase'],
                ha='center', va='bottom',
                fontsize=8,
                fontweight='bold',
                style='italic',
                color=edgecolor)

    y_text = y + height - 0.3

    # Platform name (bold, larger)
    ax.text(x, y_text, platform_data['name'],
            ha='center', va='top',
            fontsize=11,
            fontweight='bold',
            color=edgecolor)
    y_text -= 0.35

    # Mass info
    mass_text = f"{platform_data['mass']}  ({platform_data['mass_ratio']})"
    ax.text(x, y_text, mass_text,
            ha='center', va='top',
            fontsize=9,
            color='#424242')
    y_text -= 0.4

    # Create two columns for metrics
    col_width = width / 2
    left_x = x - col_width/2
    right_x = x + col_width/2

    # Left column
    ax.text(left_x, y_text, platform_data['approach'],
            ha='center', va='top',
            fontsize=8.5,
            color='#424242',
            fontweight='semibold')
    y_text -= 0.3

    ax.text(left_x, y_text, f"Steps: {platform_data['steps']}",
            ha='center', va='top',
            fontsize=8.5,
            color='#424242')

    # Right column (aligned)
    ax.text(right_x, y_text + 0.3, f"Time: {platform_data['time']}",
            ha='center', va='top',
            fontsize=8.5,
            color='#424242')

    ax.text(right_x, y_text, f"RMSE: {platform_data['rmse']}",
            ha='center', va='top',
            fontsize=8.5,
            color='#1B5E20',
            fontweight='semibold')

# Function to create transfer arrow
def create_transfer_arrow(ax, x_start, y_start, x_end, y_end, label):
    """Create a dashed transfer arrow"""
    arrow = FancyArrowPatch((x_start, y_start), (x_end, y_end),
                           arrowstyle='->,head_width=0.35,head_length=0.5',
                           color='#424242',
                           linewidth=2.5,
                           linestyle='--',
                           zorder=2)
    ax.add_patch(arrow)

    # Arrow label
    mid_y = (y_start + y_end) / 2
    ax.text(x_start + 0.8, mid_y, label,
            ha='left', va='center',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#424242', linewidth=1.5),
            color='#424242',
            fontweight='bold')

# Create boxes and arrows
for i, (platform, y_pos, (facecolor, edgecolor)) in enumerate(zip(platforms, y_positions, colors)):
    create_platform_box(ax, x_center, y_pos, box_width, box_height,
                       facecolor, edgecolor, platform)

    # Add transfer arrow (except after last box)
    if i < len(platforms) - 1:
        arrow_start_y = y_pos
        arrow_end_y = y_positions[i + 1] + box_height
        transfer_label = f"Transfer θ{i+1}\n(75% step reduction)"
        create_transfer_arrow(ax, x_center, arrow_start_y, x_center, arrow_end_y, transfer_label)

# Summary box at bottom
summary_y = 0.3
summary_height = 1.3
summary_box = FancyBboxPatch((x_center - box_width/2, summary_y), box_width, summary_height,
                             boxstyle="round,pad=0.08",
                             facecolor='#F5F5F5',
                             edgecolor='#616161',
                             linewidth=2.5,
                             zorder=1)
ax.add_patch(summary_box)

# Summary title
ax.text(x_center, summary_y + summary_height - 0.15, 'Total Training Summary',
        ha='center', va='top',
        fontsize=10,
        fontweight='bold',
        color='#424242')

# Summary content
summary_text = [
    '35,000 steps, 363 min (6.1 hrs) with transfer',
    'vs 80,000 steps, 801 min (13.4 hrs) without transfer',
    'Savings: 56.2% time, 75% steps per platform'
]
y_sum = summary_y + summary_height - 0.45
for line in summary_text:
    ax.text(x_center, y_sum, line,
            ha='center', va='top',
            fontsize=8.5,
            color='#1B5E20' if 'Savings' in line else '#424242',
            fontweight='semibold' if 'Savings' in line else 'normal')
    y_sum -= 0.25

# Title at top
ax.text(x_center, 14.7, 'Sequential Transfer Learning Across UAV Platforms',
        ha='center', va='top',
        fontsize=12,
        fontweight='bold',
        color='#424242')

# Save figure
plt.tight_layout()
plt.savefig('D:/rl_tuned_mpc/paper/figures/transfer_learning_flow.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=True)
plt.savefig('D:/rl_tuned_mpc/paper/figures/transfer_learning_flow.png',
            format='png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=False,
            facecolor='white')

print("[OK] Figure 3 (Transfer Learning Flow) generated successfully!")
print("     - PDF: D:/rl_tuned_mpc/paper/figures/transfer_learning_flow.pdf")
print("     - PNG: D:/rl_tuned_mpc/paper/figures/transfer_learning_flow.png")
