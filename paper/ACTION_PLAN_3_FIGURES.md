# Action Plan: Create 3 Figures for Paper

## Current Status
- ✅ **Figure 1** (Line 238-244): Training Results - `training_results.png` - ACTIVE
- ❌ **Figure 2** (Line 133-138): System Architecture - `system_architecture.pdf` - COMMENTED OUT
- ❌ **Figure 3** (Line 118-123): Transfer Learning Flow - `transfer_learning_flow.pdf` - COMMENTED OUT

---

## Step-by-Step Action Plan

### Step 1: Create System Architecture Figure (20-30 min)

**Open Draw.io**: https://app.diagrams.net/

**Create this diagram**:
```
┌─────────────────────────────────────────────┐
│         RL Optimizer (PPO)                  │
│                                             │
│  Input:  State s_t (29D)                    │
│          • Position/velocity errors         │
│          • Current hyperparameters          │
│                                             │
│  Output: Action a_t (17D)                   │
│          • Q weights (12D)                  │
│          • R weights (4D)                   │
│          • Horizon N (1D)                   │
└──────────────┬──────────────────────────────┘
               │
               │ θ = [Q, R, N] ∈ ℝ¹⁷
               ↓
┌─────────────────────────────────────────────┐
│         MPC Controller (CasADi)             │
│                                             │
│  Optimization:                              │
│    min J = Σ(x'Qx + u'Ru)                   │
│  Subject to:                                │
│    • Dynamics: ẋ = f(x,u)                   │
│    • Control limits                         │
│    • State bounds                           │
│                                             │
│  Output: u ∈ ℝ⁴                             │
│    [Thrust, ω_roll, ω_pitch, ω_yaw]         │
└──────────────┬──────────────────────────────┘
               │
               │ u ∈ ℝ⁴
               ↓
┌─────────────────────────────────────────────┐
│    UAV Environment (PyBullet)               │
│                                             │
│  Dynamics: ẋ = f(x, u, θ_drone)             │
│                                             │
│  State x ∈ ℝ¹²:                             │
│    • Position, Velocity                     │
│    • Euler angles, Angular rates            │
│                                             │
│  Reward: r = -||e||² - λ||u||²              │
└──────────────┬──────────────────────────────┘
               │
               │ s_t ∈ ℝ²⁹, r_t
               └─────────────────────────┐
                                         │
                  (loops back to RL Optimizer)
```

**Quick Draw.io Instructions**:
1. Create 3 rounded rectangles (General → Rounded Rectangle)
2. Color them:
   - Top box: Fill #E3F2FD (light blue), Border #1976D2 (blue)
   - Middle box: Fill #E8F5E9 (light green), Border #388E3C (green)
   - Bottom box: Fill #FFF3E0 (light orange), Border #F57C00 (orange)
3. Add arrows (use "Arrow" connector)
   - Arrow 1: Label "θ = [Q, R, N] ∈ ℝ¹⁷"
   - Arrow 2: Label "u ∈ ℝ⁴"
   - Arrow 3: Label "s_t ∈ ℝ²⁹, r_t" (make it curve back to top)
4. Add text inside each box (double-click box to edit)
5. Font: 11pt Arial, line height 1.2

**Export**:
- File → Export As → PDF
- Settings:
  - ✅ Crop to content
  - ✅ Transparent background
  - Border width: 0
- Save as: `D:\rl_tuned_mpc\paper\figures\system_architecture.pdf`

---

### Step 2: Create Transfer Learning Flow Figure (20-30 min)

**Same Draw.io window**, create new page or new file

**Create this diagram**:
```
PHASE 1
┌───────────────────────────────┐
│   Crazyflie 2.X               │
│   Mass: 0.027 kg              │
│                               │
│   Train from scratch          │
│   Steps: 20,000               │
│   Time: 200 min               │
│   RMSE: 1.34 m                │
└──────────┬────────────────────┘
           │
           │ Transfer θ₁
           │ (Save checkpoint)
           ↓
PHASE 2
┌───────────────────────────────┐
│   Racing Drone                │
│   Mass: 0.800 kg (29.6×)      │
│                               │
│   Load θ₁ → Fine-tune         │
│   Steps: 5,000 (75% ↓)        │
│   Time: 52 min                │
│   RMSE: 1.33 m                │
└──────────┬────────────────────┘
           │
           │ Transfer θ₂
           ↓
PHASE 3
┌───────────────────────────────┐
│   Generic Quad                │
│   Mass: 2.500 kg (92.6×)      │
│                               │
│   Load θ₂ → Fine-tune         │
│   Steps: 5,000 (75% ↓)        │
│   Time: 52 min                │
│   RMSE: 1.34 m                │
└──────────┬────────────────────┘
           │
           │ Transfer θ₃
           ↓
PHASE 4
┌───────────────────────────────┐
│   Heavy-Lift Hexacopter       │
│   Mass: 5.500 kg (203.7×)     │
│                               │
│   Load θ₃ → Fine-tune         │
│   Steps: 5,000 (75% ↓)        │
│   Time: 59 min                │
│   RMSE: 1.34 m                │
└───────────────────────────────┘

┌─────────────────────────────────────────┐
│ TOTAL RESULTS                           │
│ • 35,000 steps, 363 min (6.1 hrs)       │
│ • vs 80,000 steps, 801 min w/o transfer │
│ • Savings: 56.2% time, 75% steps        │
└─────────────────────────────────────────┘
```

**Quick Draw.io Instructions**:
1. Create 4 rounded rectangles for phases (vertical stack)
2. Use gradient colors (light → dark blue):
   - Phase 1: #E3F2FD
   - Phase 2: #BBDEFB
   - Phase 3: #90CAF9
   - Phase 4: #64B5F6
3. Add dashed arrows between boxes:
   - Style: Dashed, thick (2pt)
   - Labels: "Transfer θ₁", "Transfer θ₂", "Transfer θ₃"
4. Add summary box at bottom (different color, e.g., light gray #F5F5F5)
5. Font: 11pt Arial

**Export**:
- File → Export As → PDF
- Same settings as before
- Save as: `D:\rl_tuned_mpc\paper\figures\transfer_learning_flow.pdf`

---

### Step 3: Create figures folder and save PDFs

If the `figures` folder doesn't exist:
```bash
mkdir D:\rl_tuned_mpc\paper\figures
```

Save both PDFs to:
- `D:\rl_tuned_mpc\paper\figures\system_architecture.pdf`
- `D:\rl_tuned_mpc\paper\figures\transfer_learning_flow.pdf`

---

### Step 4: Uncomment figure blocks in main.tex

**Figure 2 (System Architecture)** - Line 133-138:
Remove the `%` from lines 133-138:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.85\columnwidth]{figures/system_architecture.pdf}
\caption{System architecture showing the interaction between RL optimizer (PPO), MPC controller (CasADi/IPOPT), and UAV simulation environment (PyBullet). The RL agent outputs 17-dimensional hyperparameters (Q, R, horizon) that configure the MPC controller, which generates 4-dimensional control commands for the UAV. State feedback and rewards close the learning loop.}
\label{fig:architecture}
\end{figure}
```

**Figure 3 (Transfer Learning Flow)** - Line 118-123:
Remove the `%` from lines 118-123:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.85\columnwidth]{figures/transfer_learning_flow.pdf}
\caption{Sequential transfer learning pipeline across four UAV platforms. Knowledge is transferred progressively from Crazyflie 2.X (0.027kg) through Racing Quadrotor (0.800kg) and Generic Quadrotor (2.500kg) to Heavy-Lift Hexacopter (5.500kg). Base training requires 20,000 steps while fine-tuning requires only 5,000 steps (75\% reduction), demonstrating efficient knowledge transfer across 200$\times$ mass variation.}
\label{fig:transfer_flow}
\end{figure}
```

---

### Step 5: Upload to Overleaf

1. Go to https://www.overleaf.com/latex/templates/aaai-2026-press-formatting-instructions-for-authors-using-latex/qnpmwrzmddjj
2. Click "Open as Template"
3. Upload these files:
   - `main.tex` (replace existing)
   - `references.bib` (replace existing)
   - `figures/system_architecture.pdf` (new)
   - `figures/transfer_learning_flow.pdf` (new)
   - `figures/training_results.png` (new)
4. Click "Recompile"

---

## Final Checklist

- [ ] Created `system_architecture.pdf` in Draw.io
- [ ] Created `transfer_learning_flow.pdf` in Draw.io
- [ ] Saved both PDFs to `D:\rl_tuned_mpc\paper\figures\`
- [ ] Uncommented Figure 2 block (lines 133-138) in main.tex
- [ ] Uncommented Figure 3 block (lines 118-123) in main.tex
- [ ] Uploaded all files to Overleaf
- [ ] Compiled successfully
- [ ] Verified page count ≤ 6 pages

---

## Time Estimate

- Figure 1 (System Architecture): 20-30 minutes
- Figure 2 (Transfer Learning): 20-30 minutes
- Upload to Overleaf: 5-10 minutes

**Total**: 45-70 minutes

---

## Need Help?

- Draw.io tutorial: https://www.diagrams.net/doc/
- Example diagrams are in: `THREE_FIGURES_PLAN.md`
- Overleaf guide: `QUICK_START_OVERLEAF.md`
