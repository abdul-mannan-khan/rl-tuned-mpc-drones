"""
Generate Figure 2: System Architecture Diagram
Professional publication-quality figure for AAAI paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set up publication-quality parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for editability

# Create figure
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Define colors (professional, colorblind-friendly palette)
color_rl = '#E3F2FD'      # Light blue
border_rl = '#1976D2'     # Dark blue
color_mpc = '#E8F5E9'     # Light green
border_mpc = '#388E3C'    # Dark green
color_uav = '#FFF3E0'     # Light orange
border_uav = '#F57C00'    # Dark orange

# Box parameters
box_width = 8
box_height = 3.5
x_center = 5

# Function to create a fancy box with text
def create_box(ax, x, y, width, height, facecolor, edgecolor, title, content_lines, title_size=12, content_size=9):
    """Create a professional box with title and content"""
    # Main box
    box = FancyBboxPatch((x - width/2, y), width, height,
                          boxstyle="round,pad=0.1",
                          facecolor=facecolor,
                          edgecolor=edgecolor,
                          linewidth=2.5,
                          zorder=1)
    ax.add_patch(box)

    # Title (bold)
    ax.text(x, y + height - 0.4, title,
            ha='center', va='top',
            fontsize=title_size, fontweight='bold',
            color=edgecolor)

    # Content lines
    y_text = y + height - 1.0
    for line in content_lines:
        ax.text(x, y_text, line,
                ha='center', va='top',
                fontsize=content_size,
                color='black')
        y_text -= 0.35

# Function to create arrow
def create_arrow(ax, x_start, y_start, x_end, y_end, label, label_offset=0.3):
    """Create a professional arrow with label"""
    arrow = FancyArrowPatch((x_start, y_start), (x_end, y_end),
                           arrowstyle='->,head_width=0.4,head_length=0.6',
                           color='#424242',
                           linewidth=2.5,
                           zorder=2)
    ax.add_patch(arrow)

    # Arrow label
    mid_x = (x_start + x_end) / 2
    mid_y = (y_start + y_end) / 2
    ax.text(mid_x + label_offset, mid_y, label,
            ha='left', va='center',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.9),
            color='#424242',
            fontweight='semibold')

# Box 1: RL Optimizer (PPO) - Top
y1 = 10
rl_content = [
    'Neural Network Policy πθ',
    '',
    'Input: State st ∈ ℝ²⁹',
    '  • Position/velocity errors',
    '  • Current hyperparameters',
    '',
    'Output: Action at ∈ ℝ¹⁷',
    '  • Q weights (12D)',
    '  • R weights (4D)',
    '  • Horizon N (1D)'
]
create_box(ax, x_center, y1, box_width, box_height,
           color_rl, border_rl, 'RL Optimizer (PPO)', rl_content)

# Arrow 1: RL -> MPC
create_arrow(ax, x_center, y1, x_center, y1 - 2.3,
             'θ = [Q, R, N] ∈ ℝ¹⁷', label_offset=0.5)

# Box 2: MPC Controller - Middle
y2 = 5.5
mpc_content = [
    'Optimization Problem:',
    'min J = Σ(x\'Qx + u\'Ru)',
    '',
    'Subject to:',
    '  • Dynamics: ẋ = f(x, u)',
    '  • Control limits',
    '  • State bounds',
    '',
    'Output: u ∈ ℝ⁴',
    '  [Thrust, ωroll, ωpitch, ωyaw]'
]
create_box(ax, x_center, y2, box_width, box_height,
           color_mpc, border_mpc, 'MPC Controller (CasADi/IPOPT)', mpc_content, title_size=11)

# Arrow 2: MPC -> UAV
create_arrow(ax, x_center, y2, x_center, y2 - 2.3,
             'u ∈ ℝ⁴', label_offset=0.5)

# Box 3: UAV Environment - Bottom
y3 = 1
uav_content = [
    'Physics Simulation (PyBullet)',
    '',
    'Dynamics: ẋ = f(x, u, θdrone)',
    '',
    'State x ∈ ℝ¹²:',
    '  • Position (3D), Velocity (3D)',
    '  • Euler angles (3D), Angular rates (3D)',
    '',
    'Reward: r = -||epos||² - ||evel||² - λ||u||²'
]
create_box(ax, x_center, y3, box_width, box_height,
           color_uav, border_uav, 'UAV Environment (PyBullet)', uav_content)

# Arrow 3: UAV -> RL (feedback loop)
# Create curved arrow going back to RL
from matplotlib.patches import Arc, FancyArrowPatch
# Left side up
arrow_feedback = FancyArrowPatch((x_center - box_width/2 + 0.3, y3 + box_height),
                                 (x_center - box_width/2 + 0.3, y1 + box_height/2),
                                 arrowstyle='->,head_width=0.4,head_length=0.6',
                                 color='#424242',
                                 linewidth=2.5,
                                 linestyle='--',
                                 connectionstyle="arc3,rad=0.3",
                                 zorder=2)
ax.add_patch(arrow_feedback)

# Feedback label
ax.text(0.3, (y1 + y3 + box_height) / 2, 'st ∈ ℝ²⁹, rt',
        ha='center', va='center',
        fontsize=10,
        rotation=90,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.9),
        color='#424242',
        fontweight='semibold')

# Add "Closed-Loop Control" annotation
ax.text(x_center, -0.3, 'Closed-Loop RL-MPC Framework',
        ha='center', va='top',
        fontsize=11,
        fontweight='bold',
        style='italic',
        color='#424242')

# Save figure
plt.tight_layout()
plt.savefig('D:/rl_tuned_mpc/paper/figures/system_architecture.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=True)
plt.savefig('D:/rl_tuned_mpc/paper/figures/system_architecture.png',
            format='png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=False,
            facecolor='white')

print("[OK] Figure 2 (System Architecture) generated successfully!")
print("     - PDF: D:/rl_tuned_mpc/paper/figures/system_architecture.pdf")
print("     - PNG: D:/rl_tuned_mpc/paper/figures/system_architecture.png")
