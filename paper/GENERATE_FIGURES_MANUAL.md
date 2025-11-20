# Manual Figure Generation - Step by Step Guide

## Easiest Method: PlantUML Web Editor

Since Java is not installed, here's the **simplest guaranteed method**:

### Step 1: Open PlantUML Web Editor
Go to: **https://www.plantuml.com/plantuml/uml/**

---

### Step 2: Generate Figure 2 (System Architecture)

1. **Open file**: `D:\rl_tuned_mpc\paper\figure_2_architecture.puml`
2. **Copy ENTIRE contents** (Ctrl+A, Ctrl+C)
3. **Paste into PlantUML web editor**
4. **Wait 2 seconds** - diagram will appear automatically
5. **Right-click on diagram** → "Save image as..."
6. **Save as**: `system_architecture.png` in `D:\rl_tuned_mpc\paper\figures\`

**For PDF format:**
- Click the "PNG" dropdown → Select "SVG"
- Download SVG file
- Go to https://cloudconvert.com/svg-to-pdf
- Upload SVG, convert to PDF
- Save as `system_architecture.pdf`

---

### Step 3: Generate Figure 3 (Transfer Learning Flow)

1. **Clear the web editor** (delete previous content)
2. **Open file**: `D:\rl_tuned_mpc\paper\figure_3_transfer.puml`
3. **Copy ENTIRE contents**
4. **Paste into PlantUML web editor**
5. **Wait for diagram to appear**
6. **Save image as**: `transfer_learning_flow.png`

**For PDF**: Same as above (SVG → CloudConvert → PDF)

---

### Step 4: Generate Figure 4 (PyBullet Environment)

1. **Clear the web editor**
2. **Open file**: `D:\rl_tuned_mpc\paper\figure_4_pybullet.puml`
3. **Copy ENTIRE contents**
4. **Paste into PlantUML web editor**
5. **Wait for diagram to appear**
6. **Save image as**: `pybullet_environment.png`

**For PDF**: Same as above (SVG → CloudConvert → PDF)

---

## Alternative: Use PNG files directly

The PNG files from PlantUML web editor are **high quality** (suitable for papers).

You can use them directly instead of PDF:

1. Save all 3 as PNG (steps above)
2. Update `main.tex` to reference `.png` instead of `.pdf`:

```latex
% Change from:
\includegraphics[width=\columnwidth]{figures/system_architecture.pdf}

% To:
\includegraphics[width=\columnwidth]{figures/system_architecture.png}
```

Do this for all 3 generated figures.

---

## Quick Checklist

After generation, verify you have:

- [ ] `figures/training_results.png` (already exists - Figure 1)
- [ ] `figures/system_architecture.png` or `.pdf` (Figure 2 - NEW)
- [ ] `figures/transfer_learning_flow.png` or `.pdf` (Figure 3 - NEW)
- [ ] `figures/pybullet_environment.png` or `.pdf` (Figure 4 - NEW)

---

## Expected Results

### Figure 2 should show:
- 3 colored boxes stacked vertically
- RL Optimizer (blue) at top
- MPC Controller (green) in middle
- UAV Environment (orange) at bottom
- Arrows connecting them in a loop
- Notes on the sides
- Clean, professional layout

### Figure 3 should show:
- 4 platform boxes in blue gradient
- Crazyflie → Racing → Generic → Heavy-Lift
- Transfer arrows between them
- Summary box at bottom in green
- Training metrics in each box

### Figure 4 should show:
- Central Quadrotor UAV box
- State Vector and Control Input boxes
- Physics Engine and Environment boxes
- Circular flow of arrows
- Technical details in each box

---

## Time Estimate

- Each figure: 2-3 minutes
- Total: **10 minutes maximum**

---

## Troubleshooting

**Q: Diagram doesn't appear in web editor**
- Wait 5 seconds for server to process
- Check that you copied the ENTIRE file (including @startuml and @enduml lines)
- Try refreshing the page

**Q: Image quality is poor**
- Use SVG format instead of PNG
- Convert SVG to PDF for best quality
- Or use PNG at highest resolution

**Q: Can't save as PDF**
- Save as SVG first
- Use CloudConvert.com to convert SVG → PDF
- Free, no registration required

**Q: File too large**
- PlantUML generates efficient files
- Typical size: 20-100 KB
- No compression needed

---

## Next Steps After Generation

1. **Verify all 4 figures** exist in `figures/` folder
2. **Check quality** - open each file and inspect
3. **Upload to Overleaf**:
   - All 4 figure files
   - `main.tex`
   - `references.bib`
   - `aaai24.sty`
4. **Compile and check** page count (~7 pages)
5. **Download final PDF**

---

## Support

If PlantUML web editor is slow or down:
- Alternative server: https://plantuml-editor.kkeisuke.com/
- Or: https://liveuml.com/

All work the same way - just paste the .puml code!

---

**Estimated total time**: 10-15 minutes for all 3 figures
**Difficulty**: Easy - just copy/paste and save
**Quality**: Publication-ready professional diagrams
