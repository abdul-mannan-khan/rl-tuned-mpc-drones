# Quick Start: Upload Paper to Overleaf

## TL;DR - 3 Steps to Get Your Paper on Overleaf

### Step 1: Get AAAI Template (2 minutes)
1. Go to: https://www.overleaf.com/latex/templates/aaai-2026-press-formatting-instructions-for-authors-using-latex/qnpmwrzmddjj
2. Click: **"Open as Template"**
3. You now have an Overleaf project with AAAI style file included

### Step 2: Replace Content (5 minutes)
1. In Overleaf, delete the template's `main.tex` content
2. Copy your `main.tex` from `D:\rl_tuned_mpc\paper\main.tex`
3. Paste it into Overleaf's `main.tex`
4. Upload your `references.bib` file (click "Upload")
5. Create `figures/` folder and upload `training_results.png`

### Step 3: Compile (1 minute)
1. Click **"Recompile"** button (top-right)
2. PDF generated! ✓
3. Check that figure and tables appear correctly

**Done! Your paper is now on Overleaf.**

---

## What About the pdflatex Error?

The error you saw:
```
'pdflatex' is not recognized as an internal or external command
```

**This is normal!** You don't have LaTeX installed on your computer. That's why we're using Overleaf - it has LaTeX pre-installed in the cloud.

**Solution:** Just use Overleaf. No local installation needed.

---

## Current Paper Status

✅ **What you already have:**
- main.tex (updated with real experimental data)
- references.bib (complete bibliography)
- training_results.png (4-panel experimental results figure)
- All metrics corrected (1.34±0.01m, 75% steps, 56.2% time)

⏳ **What to add (optional but recommended):**
- System architecture diagram (Draw.io)
- Transfer learning flow diagram (Draw.io)

---

## Detailed Upload Instructions

### Method 1: Manual Upload (Recommended)

1. **Open AAAI template:**
   - URL: https://www.overleaf.com/latex/templates/aaai-2026-press-formatting-instructions-for-authors-using-latex/qnpmwrzmddjj
   - Click "Open as Template"

2. **Replace main.tex:**
   - In Overleaf file list (left), click `main.tex`
   - Select ALL text (Ctrl+A) and DELETE
   - Open your local `D:\rl_tuned_mpc\paper\main.tex`
   - Copy all content (Ctrl+A, Ctrl+C)
   - Paste into Overleaf (Ctrl+V)

3. **Upload references.bib:**
   - In Overleaf, click "Upload" icon (📤)
   - Select `D:\rl_tuned_mpc\paper\references.bib`
   - Upload

4. **Upload figure:**
   - In Overleaf, click "New Folder"
   - Name: `figures`
   - Click on `figures` folder
   - Click "Upload"
   - Select `D:\rl_tuned_mpc\paper\figures\training_results.png`
   - Upload

5. **Compile:**
   - Click "Recompile" (top-right)
   - Wait 10-20 seconds
   - PDF appears on right side!

### Method 2: ZIP Upload (Faster but requires creating ZIP)

1. **Create ZIP file:**
   ```
   Right-click on D:\rl_tuned_mpc\paper\ folder
   → "Compress to ZIP file"
   → Name it: paper.zip
   ```

2. **Open AAAI template on Overleaf**

3. **Upload ZIP:**
   - Click "Upload" → "Upload .zip"
   - Select your `paper.zip`
   - Files extracted automatically

4. **Compile and check**

---

## Verifying Compilation

After clicking "Recompile", check:

✅ **Title page:**
- Paper title appears
- "Anonymous Authors" shown
- Abstract appears

✅ **Sections:**
- All sections present (Intro, Related Work, Methodology, Results, etc.)

✅ **Tables:**
- Table 1: UAV platforms (4 rows)
- Table 2: Crazyflie baseline (1 row)
- Table 3: Transfer learning results (4 platforms + total row)

✅ **Figure:**
- Figure 1 (training_results.png) appears
- Shows 4-panel visualization
- Caption appears below figure

✅ **References:**
- Bibliography appears at end
- All citations resolved (no "?" marks)

---

## Common Issues

### Issue 1: "File not found: aaai24.sty"

**Cause:** Style file missing
**Fix:** You're using AAAI 2026 template, so:
- Change `\usepackage{aaai24}` to `\usepackage{aaai26}`
- Or template should have `aaai24.sty` already

### Issue 2: Figure doesn't appear

**Cause:** Path incorrect or file not uploaded
**Fix:**
- Verify file is in `figures/` folder
- Check `\includegraphics{figures/training_results.png}` has correct path
- Try: `\includegraphics[width=0.95\textwidth]{figures/training_results}` (without extension)

### Issue 3: References not showing

**Cause:** BibTeX needs multiple passes
**Fix:**
- Click "Recompile" again (2nd time usually fixes it)
- Verify `references.bib` is uploaded

### Issue 4: Compilation timeout

**Cause:** Large files or infinite loop
**Fix:**
- Check for missing `\end{figure}` or `\end{table}`
- Reduce figure resolution if very large
- Check error log (bottom panel)

---

## Creating Draw.io Figures (Optional)

See `FIGURE_SPECIFICATIONS.md` for detailed instructions.

**Minimum recommended:**
1. System architecture (shows RL-MPC interaction)
2. Transfer learning flow (shows sequential knowledge transfer)

**How to create:**
1. Go to: https://app.diagrams.net/
2. Follow specifications in FIGURE_SPECIFICATIONS.md
3. Export as PDF (File → Export As → PDF)
4. Upload to Overleaf `figures/` folder
5. Uncomment figure in `main.tex`:
   - Remove the `%` at beginning of lines 174-179 (for system architecture)
   - Remove the `%` at beginning of lines 160-165 (for transfer learning)

---

## Timeline Estimate

| Task | Time | Status |
|------|------|--------|
| Upload to Overleaf | 5 min | Ready now |
| Verify compilation | 2 min | Ready now |
| Create 2 Draw.io figures | 1 hour | Optional |
| Upload new figures | 2 min | After creating |
| Final proofreading | 15 min | Recommended |
| **Total** | **~1.5 hours** | **Paper ready today!** |

---

## Next Steps After Overleaf Upload

1. **✅ Verify PDF looks correct**
2. **📊 (Optional) Create Draw.io figures** (see FIGURE_SPECIFICATIONS.md)
3. **📝 Final proofreading**
   - Spell check (Overleaf has built-in)
   - Check all numbers match
   - Verify figure/table references
4. **💾 Download final PDF**
5. **🚀 Submit to AAAI!**

---

## File Checklist

Before uploading, verify you have:

```
✅ main.tex                      (your paper, updated with real data)
✅ references.bib                (22 citations)
✅ figures/training_results.png  (437 KB, 4-panel results)
❓ figures/system_architecture.pdf     (optional, create with Draw.io)
❓ figures/transfer_learning_flow.pdf  (optional, create with Draw.io)
```

---

## Summary

**You're ready to upload right now!**

Your paper already has:
- Real experimental data
- Complete results figure
- All tables populated
- Proper AAAI formatting

Just upload to Overleaf template and compile. The optional Draw.io figures will enhance the paper but aren't required for a complete submission.

**Total time needed: 5-10 minutes for basic upload and verification.**

---

## Quick Links

- **Overleaf AAAI 2026 Template:** https://www.overleaf.com/latex/templates/aaai-2026-press-formatting-instructions-for-authors-using-latex/qnpmwrzmddjj
- **Draw.io (free, web-based):** https://app.diagrams.net/
- **Your local paper files:** `D:\rl_tuned_mpc\paper\`
- **Your experimental results:** `D:\rl_tuned_mpc\results\automated_pipeline_20251114_162746\`

---

**Ready? Let's go!** Open that Overleaf template and start uploading! 🚀
