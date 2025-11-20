# Draw.io Figure Specifications

Create these figures to enhance your paper's visual presentation. Each figure shows a key concept in your methodology.

---

## Figure 1: System Architecture

**Purpose:** Show how RL optimizer, MPC controller, and UAV environment interact

**Location in Paper:** Methodology section (after Problem Formulation)

**File Name:** `system_architecture.pdf` or `system_architecture.png`

### Layout:

```
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────┘

  ┌─────────────────┐
  │  RL Optimizer   │
  │   (PPO Agent)   │
  │                 │
  │  Input: State   │
  │  - Position err │
  │  - Velocity err │
  │  - MPC perf     │
  │                 │
  │  Output:        │
  │  - Q weights    │
  │  - R weights    │
  │  - Horizon N    │
  └────────┬────────┘
           │ Hyperparameters (17D)
           ↓
  ┌─────────────────┐
  │ MPC Controller  │
  │   (CasADi)      │
  │                 │
  │  Optimization:  │
  │  minimize J(x,u)│
  │  subject to:    │
  │  - Dynamics     │
  │  - Constraints  │
  └────────┬────────┘
           │ Control u (4D)
           ↓
  ┌─────────────────┐
  │ UAV Environment │
  │  (PyBullet)     │
  │                 │
  │  Dynamics:      │
  │  ẋ = f(x,u,θ)   │
  │                 │
  │  Output: State  │
  │  x (12D)        │
  └────────┬────────┘
           │ State Feedback
           ↓
      (back to RL Optimizer)
```

### Draw.io Instructions:

1. **Create 3 main boxes:**
   - RL Optimizer (top) - Blue color
   - MPC Controller (middle) - Green color
   - UAV Environment (bottom) - Orange color

2. **Add arrows:**
   - RL → MPC: "Hyperparameters θ = [Q, R, N]"
   - MPC → UAV: "Control u = [T, ω₁, ω₂, ω₃]"
   - UAV → RL: "State x + Reward r"

3. **Add details in each box:**
   - RL Optimizer: "PPO Neural Network", "17D action space"
   - MPC: "Nonlinear optimization", "IPOPT solver"
   - UAV: "12-state dynamics", "PyBullet simulation"

4. **Style:**
   - Use rounded rectangles
   - Bold arrows with labels
   - Professional color scheme (blues, greens, oranges)

---

## Figure 2: Sequential Transfer Learning Flow

**Purpose:** Visualize knowledge transfer across platforms

**Location in Paper:** Methodology section (Sequential Transfer Learning subsection)

**File Name:** `transfer_learning_flow.pdf`

### Layout:

```
┌────────────────────────────────────────────────────────────────────┐
│             SEQUENTIAL TRANSFER LEARNING PIPELINE                  │
└────────────────────────────────────────────────────────────────────┘

PHASE 1: Base Training
┌───────────────────┐
│  Crazyflie 2.X    │
│  m = 0.027 kg     │
│                   │
│  Train from       │
│  scratch          │
│  Steps: 20,000    │
│  Time: 200 min    │
└─────────┬─────────┘
          │ Transfer Policy θ₁
          ↓
PHASE 2: Fine-Tuning
┌───────────────────┐
│  Racing Drone     │
│  m = 0.800 kg     │
│                   │
│  Load θ₁          │
│  Fine-tune        │
│  Steps: 5,000     │
│  Time: 52 min     │
└─────────┬─────────┘
          │ Transfer Policy θ₂
          ↓
┌───────────────────┐
│  Generic Quad     │
│  m = 2.500 kg     │
│                   │
│  Load θ₂          │
│  Fine-tune        │
│  Steps: 5,000     │
│  Time: 52 min     │
└─────────┬─────────┘
          │ Transfer Policy θ₃
          ↓
┌───────────────────┐
│  Heavy-Lift Hex   │
│  m = 5.500 kg     │
│                   │
│  Load θ₃          │
│  Fine-tune        │
│  Steps: 5,000     │
│  Time: 59 min     │
└───────────────────┘

Total: 35,000 steps, 363 min (6.1 hours)
Without transfer: 80,000 steps, 801 min (13.4 hours)
Savings: 75% steps, 56.2% time
```

### Draw.io Instructions:

1. **Create 4 boxes vertically:**
   - Each represents one UAV platform
   - Different colors to distinguish platforms

2. **Add arrows between boxes:**
   - Label: "Transfer θ" or "Knowledge Transfer"
   - Dashed lines to show transfer

3. **In each box include:**
   - Platform name
   - Mass value
   - Training steps
   - Training time

4. **Add summary box at bottom:**
   - Total statistics
   - Comparison to baseline
   - Highlight savings

5. **Style:**
   - Use gradient colors (light to dark as mass increases)
   - Bold text for key numbers
   - Icons for checkmarks (✓) next to completed phases

---

## Figure 3: Training Pipeline Flowchart

**Purpose:** Show the automated training pipeline logic

**Location in Paper:** Experimental Setup or Methodology

**File Name:** `training_pipeline.pdf`

### Layout:

```
┌─────────────────────────────────────────────────────────────────┐
│               AUTOMATED TRAINING PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

    START
      │
      ↓
┌─────────────────┐
│ Initialize      │
│ Drone Platform  │
│ (PyBullet env)  │
└────────┬────────┘
         │
         ↓
    ┌────────────┐
    │ First      │  YES
    │ Platform?  ├──────→ ┌──────────────────┐
    └────┬───────┘        │ Train from       │
         │ NO             │ Scratch          │
         ↓                │ (20,000 steps)   │
┌────────────────┐        └────────┬─────────┘
│ Load Previous  │                 │
│ Policy θ_{i-1} │                 │
└────────┬───────┘                 │
         │                         │
         ↓                         │
┌────────────────┐                 │
│ Fine-Tune      │←────────────────┘
│ (5,000 steps)  │
└────────┬───────┘
         │
         ↓
┌────────────────┐
│ Save Model     │
│ PPO_{drone}.zip│
└────────┬───────┘
         │
         ↓
┌────────────────┐
│ Checkpoint     │
│ State          │
└────────┬───────┘
         │
         ↓
    ┌────────────┐
    │ More       │  YES
    │ Platforms? ├──────→ (back to Initialize)
    └────┬───────┘
         │ NO
         ↓
┌─────────────────┐
│ Generate Report │
│ & Visualizations│
└────────┬────────┘
         │
         ↓
      END
```

### Draw.io Instructions:

1. **Use flowchart symbols:**
   - Ovals for START/END
   - Rectangles for processes
   - Diamonds for decisions

2. **Color coding:**
   - Green: Training steps
   - Blue: Data operations
   - Yellow: Decision points

3. **Add icons:**
   - 💾 Save operations
   - 🔄 Loops
   - ✓ Checkpoints

---

## Figure 4: MPC-RL Integration Detail

**Purpose:** Show how RL action space maps to MPC hyperparameters

**Location in Paper:** Problem Formulation or Methodology

**File Name:** `mpc_rl_integration.pdf`

### Layout:

```
┌──────────────────────────────────────────────────────────────┐
│              RL-MPC HYPERPARAMETER MAPPING                    │
└──────────────────────────────────────────────────────────────┘

RL Action Space (17 dimensions)
┌────────────────────────────────────────┐
│  a = [a₁, a₂, ..., a₁₇] ∈ ℝ¹⁷         │
│  Continuous, normalized [-1, 1]        │
└──────────────────┬─────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
       ↓                       ↓
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Q Matrix     │      │ R Matrix     │      │ Horizon      │
│ (12 dims)    │      │ (4 dims)     │      │ (1 dim)      │
│              │      │              │      │              │
│ State        │      │ Control      │      │ Prediction   │
│ Weights      │      │ Weights      │      │ Steps        │
│              │      │              │      │              │
│ q₁...q₁₂     │      │ r₁...r₄      │      │ N ∈ [5,20]   │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                             │
                             ↓
                   ┌─────────────────┐
                   │  MPC Cost J(x,u)│
                   │                 │
                   │  J = Σ(x'Qx +   │
                   │      u'Ru)      │
                   └─────────────────┘
```

### Draw.io Instructions:

1. **Top box:** RL action vector (17D)
2. **Three middle boxes:** Q matrix, R matrix, Horizon
3. **Bottom box:** MPC cost function
4. **Arrows:** Show mapping from actions to hyperparameters
5. **Math notation:** Use LaTeX-style text for equations

---

## Figure 5: Reward Function Components

**Purpose:** Visualize multi-objective reward engineering

**Location in Paper:** Methodology (Reward Engineering subsection)

**File Name:** `reward_structure.pdf`

### Layout:

```
┌──────────────────────────────────────────────────────────┐
│               MULTI-OBJECTIVE REWARD FUNCTION             │
└──────────────────────────────────────────────────────────┘

                    r_total
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ↓              ↓              ↓
┌───────────────┐ ┌──────────┐ ┌────────────┐
│   Tracking    │ │ Control  │ │  Penalty   │
│   Accuracy    │ │  Effort  │ │   Terms    │
└───────┬───────┘ └────┬─────┘ └─────┬──────┘
        │              │              │
        ↓              ↓              ↓
 -10.0‖eₚₒₛ‖    -0.01‖u‖      -5.0·𝟙(overshoot)
        │              │              │
        ↓              ↓              ↓
   Position       Smoothness      Stability
     Error         Control        Constraint
```

### Draw.io Instructions:

1. **Tree structure:** Root node = total reward
2. **Three branches:** Tracking, Control, Penalty
3. **Leaf nodes:** Specific reward components
4. **Add weights:** Show coefficients (-10.0, -0.01, -5.0)
5. **Color coding:**
   - Green: Positive contributions (accuracy)
   - Yellow: Efficiency (control effort)
   - Red: Penalties (violations)

---

## General Draw.io Tips

### How to Create Figures:

1. **Go to:** https://app.diagrams.net/ (free, web-based)
2. **Choose:** "Device" (save locally) or "Google Drive"
3. **Select template:** "Blank Diagram" or "Flowchart"
4. **Use shapes from left panel:**
   - Basic Shapes → Rectangles, Ovals
   - Flowchart → Decision diamonds
   - Arrows & Connectors

### Styling Guidelines:

- **Font:** Arial or Helvetica, size 10-12pt
- **Colors:** Professional palette (blues, greens, avoid bright colors)
- **Line width:** 2pt for main arrows, 1pt for details
- **Alignment:** Use "Arrange → Align" for clean layouts
- **Spacing:** Consistent padding between elements

### Export Settings:

1. **File → Export As → PDF**
2. **Settings:**
   - ✅ Crop to content
   - ✅ Transparent background
   - Resolution: 100% (will be vector)
   - Border width: 0

3. **Or PNG:**
   - Resolution: 300 DPI minimum
   - Width: 2000-3000 pixels
   - ✅ Transparent background

---

## Quick Reference: Which Figures to Create

**Minimum (2 figures):**
1. ✅ System Architecture (most important)
2. ✅ Transfer Learning Flow (shows key contribution)

**Recommended (add 1-2 more):**
3. Training Pipeline Flowchart
4. MPC-RL Integration Detail

**Optional:**
5. Reward Structure (if space permits)

---

## Time Estimate

- **Per figure:** 20-30 minutes
- **Total for 2 figures:** 1 hour
- **Total for 4 figures:** 2 hours

**Priority Order:**
1. System Architecture (essential)
2. Transfer Learning Flow (key contribution)
3. Training Pipeline (shows automation)
4. MPC-RL Integration (technical depth)

---

See OVERLEAF_UPLOAD_GUIDE.md for instructions on uploading these figures once created.
