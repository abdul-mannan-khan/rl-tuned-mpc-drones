"""
Generate Figure 4: PyBullet Simulation Environment
Shows UAV in simulated environment with labels
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Wedge
import matplotlib.patches as patches
import numpy as np

# Set up publication-quality parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['pdf.fonttype'] = 42

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(5, 6.7, 'PyBullet UAV Simulation Environment',
        ha='center', va='top', fontsize=13, fontweight='bold', color='#424242')

# Draw ground plane
ground = Rectangle((0, 0.5), 10, 0.1, facecolor='#8D6E63', edgecolor='#5D4037', linewidth=2)
ax.add_patch(ground)
ax.text(5, 0.3, 'Ground Plane (Collision Detection)', ha='center', va='top',
        fontsize=9, style='italic', color='#5D4037')

# Draw 3D grid to show space
grid_color = '#E0E0E0'
for x in np.linspace(1, 9, 9):
    ax.plot([x, x], [0.6, 6], color=grid_color, linewidth=0.5, alpha=0.3, linestyle='--')
for y in np.linspace(1, 6, 6):
    ax.plot([1, 9], [y, y], color=grid_color, linewidth=0.5, alpha=0.3, linestyle='--')

# Draw quadrotor (center of scene)
quad_x, quad_y = 5, 3.5

# Quadrotor body
body_size = 0.4
body = Rectangle((quad_x - body_size/2, quad_y - body_size/2),
                 body_size, body_size,
                 facecolor='#1976D2', edgecolor='#0D47A1', linewidth=2.5)
ax.add_patch(body)

# Quadrotor arms
arm_length = 0.8
arm_width = 0.08
# Arm 1 (NE)
arm1 = Rectangle((quad_x - arm_width/2, quad_y - arm_width/2),
                 arm_length, arm_width, angle=45,
                 facecolor='#424242', edgecolor='#212121', linewidth=1.5)
ax.add_patch(arm1)
# Arm 2 (NW)
arm2 = Rectangle((quad_x - arm_length, quad_y - arm_width/2),
                 arm_length, arm_width, angle=-45,
                 facecolor='#424242', edgecolor='#212121', linewidth=1.5)
ax.add_patch(arm2)

# Motors/Propellers
motor_positions = [
    (quad_x + 0.5, quad_y + 0.5),   # Front-right
    (quad_x - 0.5, quad_y + 0.5),   # Front-left
    (quad_x + 0.5, quad_y - 0.5),   # Back-right
    (quad_x - 0.5, quad_y - 0.5),   # Back-left
]

for i, (mx, my) in enumerate(motor_positions):
    # Motor
    motor = Circle((mx, my), 0.12, facecolor='#616161', edgecolor='#212121', linewidth=1.5)
    ax.add_patch(motor)

    # Propeller (spinning effect)
    for angle in [0, 90]:
        prop = patches.Ellipse((mx, my), 0.35, 0.08, angle=angle,
                              facecolor='#90CAF9', edgecolor='#1976D2',
                              linewidth=1, alpha=0.6)
        ax.add_patch(prop)

    # Thrust arrows
    thrust = FancyArrowPatch((mx, my), (mx, my + 0.4),
                           arrowstyle='->,head_width=0.15,head_length=0.2',
                           color='#FF6F00', linewidth=2, alpha=0.8)
    ax.add_patch(thrust)

# Add orientation axes on UAV
axis_len = 0.3
# X-axis (red)
ax.arrow(quad_x, quad_y, axis_len, 0, head_width=0.08, head_length=0.08,
         fc='red', ec='red', linewidth=2, alpha=0.9)
ax.text(quad_x + axis_len + 0.15, quad_y, 'x', fontsize=10, fontweight='bold', color='red')

# Y-axis (green)
ax.arrow(quad_x, quad_y, 0, axis_len, head_width=0.08, head_length=0.08,
         fc='green', ec='green', linewidth=2, alpha=0.9)
ax.text(quad_x, quad_y + axis_len + 0.15, 'y', fontsize=10, fontweight='bold', color='green')

# Z-axis (blue) - showing as diagonal since 2D projection
ax.arrow(quad_x, quad_y, -0.15, 0.15, head_width=0.08, head_length=0.08,
         fc='blue', ec='blue', linewidth=2, alpha=0.9)
ax.text(quad_x - 0.3, quad_y + 0.3, 'z', fontsize=10, fontweight='bold', color='blue')

# Reference trajectory (dashed path)
trajectory_x = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
trajectory_y = np.array([2.0, 2.8, 3.5, 3.8, 3.5, 2.8, 2.0, 1.5])
ax.plot(trajectory_x, trajectory_y, 'r--', linewidth=2, alpha=0.7, label='Reference Trajectory')

# Target waypoint
target_x, target_y = 6.5, 2.8
target = Circle((target_x, target_y), 0.15, facecolor='red', edgecolor='darkred',
                linewidth=2, alpha=0.7)
ax.add_patch(target)
ax.text(target_x, target_y - 0.4, 'Target', ha='center', fontsize=9,
        fontweight='bold', color='darkred')

# Current position marker
current = Circle((quad_x, quad_y), 0.1, facecolor='lime', edgecolor='darkgreen',
                linewidth=2)
ax.add_patch(current)

# State vector annotation
state_box_x, state_box_y = 0.3, 5.5
state_box = FancyBboxPatch((state_box_x, state_box_y), 2.5, 1,
                          boxstyle="round,pad=0.1",
                          facecolor='#FFF3E0',
                          edgecolor='#F57C00',
                          linewidth=2)
ax.add_patch(state_box)
ax.text(state_box_x + 1.25, state_box_y + 0.85, 'State Vector x(t)',
        ha='center', fontsize=9, fontweight='bold', color='#E65100')
state_text = [
    'Position: [x, y, z]',
    'Velocity: [vx, vy, vz]',
    'Angles: [φ, θ, ψ]',
    'Rates: [p, q, r]'
]
y_pos = state_box_y + 0.65
for line in state_text:
    ax.text(state_box_x + 1.25, y_pos, line, ha='center', fontsize=7.5, color='#424242')
    y_pos -= 0.18

# Control input annotation
control_box_x, control_box_y = 7.2, 5.5
control_box = FancyBboxPatch((control_box_x, control_box_y), 2.5, 1,
                            boxstyle="round,pad=0.1",
                            facecolor='#E8F5E9',
                            edgecolor='#388E3C',
                            linewidth=2)
ax.add_patch(control_box)
ax.text(control_box_x + 1.25, control_box_y + 0.85, 'Control Input u(t)',
        ha='center', fontsize=9, fontweight='bold', color='#1B5E20')
control_text = [
    'Thrust: T',
    'Roll rate: ω_φ',
    'Pitch rate: ω_θ',
    'Yaw rate: ω_ψ'
]
y_pos = control_box_y + 0.65
for line in control_text:
    ax.text(control_box_x + 1.25, y_pos, line, ha='center', fontsize=7.5, color='#424242')
    y_pos -= 0.18

# Physics engine annotation
physics_box_x, physics_box_y = 0.3, 4
physics_box = FancyBboxPatch((physics_box_x, physics_box_y), 2.5, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor='#E3F2FD',
                            edgecolor='#1976D2',
                            linewidth=2)
ax.add_patch(physics_box)
ax.text(physics_box_x + 1.25, physics_box_y + 0.65, 'PyBullet Physics',
        ha='center', fontsize=9, fontweight='bold', color='#0D47A1')
physics_text = [
    'RK4 Integration (Δt=1ms)',
    'Rigid Body Dynamics',
    'Motor Thrust Model'
]
y_pos = physics_box_y + 0.45
for line in physics_text:
    ax.text(physics_box_x + 1.25, y_pos, line, ha='center', fontsize=7.5, color='#424242')
    y_pos -= 0.18

# Simulation features
features_box_x, features_box_y = 7.2, 4
features_box = FancyBboxPatch((features_box_x, features_box_y), 2.5, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor='#F3E5F5',
                             edgecolor='#7B1FA2',
                             linewidth=2)
ax.add_patch(features_box)
ax.text(features_box_x + 1.25, features_box_y + 0.65, 'Environment Features',
        ha='center', fontsize=9, fontweight='bold', color='#4A148C')
features_text = [
    'Ground Effect Modeling',
    'Aerodynamic Drag',
    'Collision Detection'
]
y_pos = features_box_y + 0.45
for line in features_text:
    ax.text(features_box_x + 1.25, y_pos, line, ha='center', fontsize=7.5, color='#424242')
    y_pos -= 0.18

# Add label to quadrotor
ax.text(quad_x, quad_y - 1, 'Quadrotor UAV\n(12D State, 4D Control)',
        ha='center', fontsize=9, fontweight='bold', color='#0D47A1',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1976D2', linewidth=1.5))

# Add simulation info at bottom
info_text = 'High-Fidelity 3D Simulation | Real-Time Physics | 240Hz Update Rate'
ax.text(5, 0.05, info_text, ha='center', fontsize=8.5, style='italic',
        color='#616161', fontweight='semibold')

# Save figure
plt.tight_layout()
plt.savefig('D:/rl_tuned_mpc/paper/figures/pybullet_environment.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=True)
plt.savefig('D:/rl_tuned_mpc/paper/figures/pybullet_environment.png',
            format='png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=False,
            facecolor='white')

print("[OK] Figure 4 (PyBullet Environment) generated successfully!")
print("     - PDF: D:/rl_tuned_mpc/paper/figures/pybullet_environment.pdf")
print("     - PNG: D:/rl_tuned_mpc/paper/figures/pybullet_environment.png")
