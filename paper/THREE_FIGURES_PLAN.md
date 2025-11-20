# Three Essential Figures for the Paper

## Current Status
- ✅ **Figure 1**: Training Results (already included) - `training_results.png`
- ❌ **Figure 2**: System Architecture - TO CREATE
- ❌ **Figure 3**: Sequential Transfer Learning Flow - TO CREATE

---

## Figure 2: System Architecture Diagram

**Purpose**: Show how RL, MPC, and UAV environment interact in closed loop

**File to create**: `figures/system_architecture.pdf`

**What to show**:
```
┌─────────────────────────────────────────────┐
│         RL Optimizer (PPO)                  │
│  Neural Network Policy π_θ                  │
│                                             │
│  Input:  State s_t (29D)                    │
│          - Position/velocity errors         │
│          - Current hyperparameters          │
│                                             │
│  Output: Action a_t (17D)                   │
│          - Q weights (12D)                  │
│          - R weights (4D)                   │
│          - Horizon N (1D)                   │
└──────────────┬──────────────────────────────┘
               │
               │ Hyperparameters θ
               ↓
┌─────────────────────────────────────────────┐
│         MPC Controller (CasADi)             │
│                                             │
│  Optimization:                              │
│  min J = Σ(x'Qx + u'Ru)                     │
│                                             │
│  Subject to:                                │
│  - Dynamics constraints                     │
│  - Control limits                           │
│  - State bounds                             │
│                                             │
│  Output: Optimal control u (4D)             │
│          [Thrust, ω_roll, ω_pitch, ω_yaw]   │
└──────────────┬──────────────────────────────┘
               │
               │ Control u
               ↓
┌─────────────────────────────────────────────┐
│    UAV Environment (PyBullet Physics)       │
│                                             │
│  Dynamics: ẋ = f(x, u, θ_drone)             │
│                                             │
│  State x (12D):                             │
│  - Position (3D)                            │
│  - Velocity (3D)                            │
│  - Euler angles (3D)                        │
│  - Angular rates (3D)                       │
│                                             │
│  Reward: r = -||e_pos|| - ||e_vel|| - ...   │
└──────────────┬──────────────────────────────┘
               │
               │ State x + Reward r
               └──────────────┐
                              │
               (feedback loop back to RL Optimizer)
```

**Draw.io Steps**:
1. Create 3 large rounded rectangles (vertical stack)
2. Use professional colors:
   - RL Optimizer: Light blue (#E3F2FD)
   - MPC Controller: Light green (#E8F5E9)
   - UAV Environment: Light orange (#FFF3E0)
3. Add thick arrows between boxes:
   - RL → MPC: Label "θ = [Q, R, N]"
   - MPC → UAV: Label "u ∈ ℝ⁴"
   - UAV → RL: Label "s_t ∈ ℝ²⁹, r_t"
4. Add text inside each box (see diagram above)
5. Font: 11pt Arial
6. Export as PDF with transparent background

---

## Figure 3: Sequential Transfer Learning Flow

**Purpose**: Show the key contribution - progressive knowledge transfer across 4 platforms

**File to create**: `figures/transfer_learning_flow.pdf`

**What to show**:
```
PHASE 1: Base Training
┌────────────────────────────┐
│   Crazyflie 2.X (0.027 kg) │
│                            │
│   Train from scratch       │
│   Steps: 20,000            │
│   Time: 200 min            │
│   Final RMSE: 1.34m        │
└──────────┬─────────────────┘
           │
           │ Transfer θ₁
           │ (75% step reduction)
           ↓
PHASE 2: Fine-tuning
┌────────────────────────────┐
│   Racing Drone (0.800 kg)  │
│   29.6× heavier            │
│                            │
│   Load θ₁ → Fine-tune      │
│   Steps: 5,000 (25%)       │
│   Time: 52 min             │
│   Final RMSE: 1.33m        │
└──────────┬─────────────────┘
           │
           │ Transfer θ₂
           │ (75% step reduction)
           ↓
PHASE 3: Fine-tuning
┌────────────────────────────┐
│   Generic Quad (2.500 kg)  │
│   92.6× heavier            │
│                            │
│   Load θ₂ → Fine-tune      │
│   Steps: 5,000 (25%)       │
│   Time: 52 min             │
│   Final RMSE: 1.34m        │
└──────────┬─────────────────┘
           │
           │ Transfer θ₃
           │ (75% step reduction)
           ↓
PHASE 4: Fine-tuning
┌────────────────────────────┐
│  Heavy-Lift Hex (5.500 kg) │
│  203.7× heavier            │
│                            │
│   Load θ₃ → Fine-tune      │
│   Steps: 5,000 (25%)       │
│   Time: 59 min             │
│   Final RMSE: 1.34m        │
└────────────────────────────┘

Total: 35,000 steps, 363 min (6.1 hrs)
vs 80,000 steps, 801 min (13.4 hrs) without transfer
Savings: 56.2% time, 75% steps per platform
```

**Draw.io Steps**:
1. Create 4 rounded rectangles (vertical stack)
2. Use gradient colors (light blue → darker blue for each level)
3. Add dashed arrows between boxes with labels
4. Each box contains:
   - Platform name and mass (bold)
   - Training approach
   - Steps count
   - Time
   - Final performance
5. Add summary box at bottom with total metrics
6. Export as PDF with transparent background

---

## Figure 4 (Optional - If Space Allows): Trajectory Tracking Example

**Purpose**: Visual example of UAV following reference trajectory

**File to create**: `figures/trajectory_example.pdf` OR extract from training_results.png

**What to show**:
- 3D plot of reference trajectory (dashed line)
- Actual UAV trajectory (solid line)
- Start and end points marked
- Shows tight tracking performance

**Alternative**: If training_results.png has 4 panels, you could:
1. Keep panels 1-2 as Figure 1
2. Use panels 3-4 as separate figures
3. This gives you 3 figures without creating new ones!

---

## Quick Timeline

**Option A: Create 2 new figures in Draw.io**
- Figure 2 (System Architecture): 20-30 minutes
- Figure 3 (Transfer Learning Flow): 20-30 minutes
- **Total time**: 40-60 minutes

**Option B: Split existing figure + create 1 new figure**
- If training_results.png has multiple panels, split it
- Create only Figure 2 (System Architecture): 20-30 minutes
- **Total time**: 30-40 minutes

---

## Draw.io Quick Start

1. **Open**: https://app.diagrams.net/
2. **Choose**: Device (saves locally)
3. **Template**: Blank Diagram
4. **Create diagrams** following specs above
5. **Export**: File → Export As → PDF
   - ✅ Crop to content
   - ✅ Transparent background
   - Border: 0
6. **Save**: To `D:\rl_tuned_mpc\paper\figures\` folder

---

## After Creating Figures

Upload these files to Overleaf in the `figures/` folder:
- `system_architecture.pdf`
- `transfer_learning_flow.pdf`
- `training_results.png` (already exists)

Then uncomment the figure blocks in main.tex (I've already prepared them with proper captions).

---

**Recommendation**: Start with Figure 2 (System Architecture) - it's the most important and shows your core contribution clearly.
