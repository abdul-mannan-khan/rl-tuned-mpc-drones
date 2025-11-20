# Figures to Create NOW - Priority Order

## Current Status
- ✅ Figure 1: Training Results (4-panel) - `training_results.png` (DONE)
- ❌ Need 2-3 more figures for better visual presentation

---

## Priority 1: System Architecture (ESSENTIAL)

**Why:** Shows the core contribution - how RL, MPC, and UAV interact

**What to show:**
```
┌─────────────────────────────────────────────┐
│         RL Optimizer (PPO)                  │
│  Input: State + Tracking Error              │
│  Output: Hyperparameters [Q, R, N]          │
│         (17 dimensions)                     │
└──────────────┬──────────────────────────────┘
               │
               │ θ = [q₁...q₁₂, r₁...r₄, N]
               ↓
┌─────────────────────────────────────────────┐
│         MPC Controller (CasADi)             │
│  Solve: min J = Σ(x'Qx + u'Ru)              │
│  Subject to: dynamics, constraints          │
│  Output: Control u [T, ω₁, ω₂, ω₃]          │
└──────────────┬──────────────────────────────┘
               │
               │ u (4D control)
               ↓
┌─────────────────────────────────────────────┐
│         UAV Environment (PyBullet)          │
│  Dynamics: ẋ = f(x, u, θ_drone)             │
│  State: x (12D) - pos, vel, angles          │
│  Reward: r(tracking_error, control_effort)  │
└──────────────┬──────────────────────────────┘
               │
               │ State x + Reward r
               ↓
        (back to RL Optimizer)
```

**Draw.io Steps:**
1. Open: https://app.diagrams.net/
2. Create 3 rounded rectangles (vertical stack)
3. Label: "RL Optimizer (PPO)", "MPC Controller (CasADi)", "UAV Environment (PyBullet)"
4. Add arrows with labels between them
5. Colors: Blue → Green → Orange
6. Export as PDF: `system_architecture.pdf`
7. Upload to Overleaf `figures/` folder

**Time:** 20-30 minutes

---

## Priority 2: Sequential Transfer Learning Flow (ESSENTIAL)

**Why:** This is your KEY CONTRIBUTION - shows knowledge transfer

**What to show:**
```
PHASE 1: Base Training
┌──────────────────────┐
│   Crazyflie 2.X      │
│   Mass: 0.027 kg     │
│   ━━━━━━━━━━━━━━━━━  │
│   Train from scratch │
│   Steps: 20,000      │
│   Time: 200 min      │
└──────────┬───────────┘
           │ Transfer θ₁
           ↓
┌──────────────────────┐
│   Racing Drone       │
│   Mass: 0.800 kg     │
│   ━━━━━━━━━━━━━━━━━  │
│   Load θ₁ → Fine-tune│
│   Steps: 5,000       │
│   Time: 52 min       │
│   Savings: 75%       │
└──────────┬───────────┘
           │ Transfer θ₂
           ↓
┌──────────────────────┐
│   Generic Quad       │
│   Mass: 2.500 kg     │
│   ━━━━━━━━━━━━━━━━━  │
│   Load θ₂ → Fine-tune│
│   Steps: 5,000       │
│   Time: 52 min       │
│   Savings: 75%       │
└──────────┬───────────┘
           │ Transfer θ₃
           ↓
┌──────────────────────┐
│   Heavy-Lift Hex     │
│   Mass: 5.500 kg     │
│   ━━━━━━━━━━━━━━━━━  │
│   Load θ₃ → Fine-tune│
│   Steps: 5,000       │
│   Time: 59 min       │
│   Savings: 75%       │
└──────────────────────┘

Total: 35,000 steps, 363 min
vs 80,000 steps, 801 min (without transfer)
```

**Draw.io Steps:**
1. Create 4 boxes vertically
2. Use gradient colors (light blue → dark blue)
3. Add dashed arrows labeled "Transfer θ"
4. Include key metrics in each box
5. Add summary box at bottom
6. Export as PDF: `transfer_learning_flow.pdf`

**Time:** 20-30 minutes

---

## Priority 3: MPC Hyperparameter Mapping (GOOD TO HAVE)

**Why:** Shows technical depth - how 17D action space maps to MPC

**What to show:**
```
RL Action Space (17D continuous)
        a ∈ [-1, 1]¹⁷
              │
    ┌─────────┴─────────┐
    │                   │
    ↓                   ↓
┌─────────┐      ┌─────────┐      ┌─────────┐
│ Q Matrix│      │ R Matrix│      │ Horizon │
│ (12D)   │      │ (4D)    │      │ (1D)    │
│         │      │         │      │         │
│ State   │      │ Control │      │ N ∈     │
│ Weights │      │ Weights │      │ [5,20]  │
└────┬────┘      └────┬────┘      └────┬────┘
     │                │                │
     └────────────────┴────────────────┘
                      │
                      ↓
            MPC Cost Function
            J = Σ(x'Qx + u'Ru)
```

**Draw.io Steps:**
1. Top box: RL action vector
2. Three middle boxes: Q, R, N
3. Bottom box: Cost function
4. Arrows showing mapping
5. Export as PDF: `hyperparameter_mapping.pdf`

**Time:** 15-20 minutes

---

## Quick Draw.io Tutorial

### Setup:
1. Go to: https://app.diagrams.net/
2. Choose "Device" (saves to your computer)
3. Select "Blank Diagram"

### Creating Boxes:
- Left panel → Basic Shapes → Rectangle/Rounded Rectangle
- Drag onto canvas
- Double-click to add text
- Right panel to change colors, borders

### Creating Arrows:
- Left panel → Arrows → Block Arrow or Line
- Drag from one box to another
- Double-click arrow to add label

### Styling:
- **Font:** 11-12pt Arial/Helvetica
- **Colors:** Professional blues/greens/oranges
- **Line width:** 2pt for main arrows
- **Alignment:** Use "Arrange → Align" for clean layouts

### Exporting:
1. File → Export As → PDF
2. Settings:
   - ✅ Crop to content
   - ✅ Transparent background
   - Border: 0
3. Save

---

## After Creating Figures

### 1. Upload to Overleaf:
- Create `figures/` folder (if not exists)
- Upload your PDFs

### 2. Uncomment in main.tex:

**For System Architecture (around line 174):**
Remove the `%` from:
```latex
%\begin{figure}[t]
%\centering
%\includegraphics[width=0.85\columnwidth]{figures/system_architecture.pdf}
%\caption{System architecture...}
%\label{fig:architecture}
%\end{figure}
```

**For Transfer Learning Flow (around line 160):**
Remove the `%` from:
```latex
%\begin{figure}[t]
%\centering
%\includegraphics[width=0.85\columnwidth]{figures/transfer_learning_flow.pdf}
%\caption{Sequential transfer learning...}
%\label{fig:transfer_flow}
%\end{figure}
```

### 3. Recompile in Overleaf

---

## Recommended Figure Set

For a 6-page paper, include:

1. ✅ **Figure 1:** Training Results (4-panel) - DONE
2. ❗ **Figure 2:** System Architecture - CREATE THIS (20 min)
3. ❗ **Figure 3:** Transfer Learning Flow - CREATE THIS (20 min)
4. ❓ **Figure 4 (optional):** Hyperparameter Mapping (15 min)

**Total time:** 40-60 minutes for 2 essential figures

---

## Alternative: Use Your Experimental Figure

If short on time, you can extract panels from `training_results.png`:

- **Panel 2 (Training Efficiency)** could be Figure 2
- **Panel 1 (Performance)** could be Figure 3

But creating new figures will look more professional!

---

See full specifications in: `FIGURE_SPECIFICATIONS.md`
