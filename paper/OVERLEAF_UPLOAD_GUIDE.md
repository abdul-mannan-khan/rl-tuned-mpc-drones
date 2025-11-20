# How to Upload Your Paper to Overleaf

## Step-by-Step Instructions

### Step 1: Prepare Files for Upload

You need to upload these files from the `paper/` directory:

**Required Files:**
```
✅ main.tex                     (your paper content)
✅ references.bib               (bibliography)
✅ figures/training_results.png (experimental results)
```

**Optional but Recommended:**
```
- Any additional figures you create in Draw.io
```

### Step 2: Create Overleaf Project

1. **Go to Overleaf:**
   - Visit: https://www.overleaf.com/
   - Log in or create free account

2. **Start from AAAI 2026 Template:**
   - Click: "New Project"
   - Select: "From Template"
   - Search for: "AAAI 2026"
   - Or use direct link: https://www.overleaf.com/latex/templates/aaai-2026-press-formatting-instructions-for-authors-using-latex/qnpmwrzmddjj
   - Click: "Open as Template"

### Step 3: Replace Template Content

**Option A: Upload Individual Files (Recommended)**

1. In the Overleaf project, **delete** the template's `main.tex` content
2. Click "Upload" button (top-left)
3. Upload your files one by one:
   - Upload `main.tex` (will replace template version)
   - Upload `references.bib`
   - Create a `figures/` folder (click "New Folder")
   - Upload `training_results.png` into `figures/` folder

**Option B: Upload as ZIP**

1. On your computer, create a ZIP file:
   ```
   paper_upload.zip
   ├── main.tex
   ├── references.bib
   └── figures/
       └── training_results.png
   ```

2. In Overleaf:
   - Click "Upload" → "Upload .zip"
   - Select your ZIP file
   - Files will be extracted automatically

### Step 4: Update Document Class

The AAAI 2026 template uses slightly different packages. You may need to make small adjustments:

**Check the first few lines of your main.tex:**
```latex
\documentclass[letterpaper]{article}
\usepackage{aaai24}  % ← May need to change to aaai26
```

**If using AAAI 2026 template, change to:**
```latex
\documentclass[letterpaper]{article}
\usepackage{aaai26}  % Updated for 2026
```

**OR** keep your current setup and the template's `aaai24.sty` should work fine.

### Step 5: Verify Compilation

1. Click **"Recompile"** button (top-right)
2. PDF should generate automatically
3. Check that:
   - ✅ All sections appear
   - ✅ Figure shows up (training_results.png)
   - ✅ Tables are formatted correctly
   - ✅ References appear at the end

### Step 6: Add Additional Figures

As you create Draw.io figures:

1. Export from Draw.io as **PDF** (preferred) or **PNG** (300 DPI minimum)
2. In Overleaf, click "Upload"
3. Upload figure to `figures/` folder
4. Reference in LaTeX:
   ```latex
   \begin{figure}[t]
   \centering
   \includegraphics[width=0.8\columnwidth]{figures/your_figure.pdf}
   \caption{Your caption here}
   \label{fig:your_label}
   \end{figure}
   ```

---

## Common Issues and Solutions

### Issue: "File not found: aaai24.sty"

**Solution:** The Overleaf template includes this. If missing:
- Use the AAAI template: https://www.overleaf.com/latex/templates/aaai-2026-press-formatting-instructions-for-authors-using-latex/qnpmwrzmddjj
- Or change to `\usepackage{aaai26}`

### Issue: Figure doesn't appear

**Solution:**
- Check figure is uploaded to `figures/` folder
- Verify path in `\includegraphics{figures/your_file.png}`
- Make sure file extension matches (`.png` vs `.pdf`)

### Issue: References not showing

**Solution:**
- Ensure `references.bib` is uploaded
- Check line: `\bibliography{references}` (no `.bib` extension)
- Click "Recompile" twice (BibTeX needs two passes)

### Issue: Compilation errors

**Solution:**
- Check the error log (bottom of screen)
- Most common: missing `$` for math mode
- Or: undefined references (need to compile twice)

---

## File Structure in Overleaf

After upload, your project should look like:

```
📁 Your AAAI Paper Project
├── 📄 main.tex                    (your paper)
├── 📄 references.bib              (bibliography)
├── 📄 aaai24.sty or aaai26.sty   (provided by template)
└── 📁 figures/
    ├── 🖼️ training_results.png    (experimental results)
    ├── 🖼️ system_architecture.pdf (to be added)
    ├── 🖼️ methodology_flow.pdf    (to be added)
    └── 🖼️ transfer_learning.pdf   (to be added)
```

---

## Tips for Using Overleaf

1. **Auto-compile:** Enable "Auto Compile" for real-time preview
2. **Collaboration:** Share link with co-authors (Project → Share)
3. **Version history:** Access via "History" button (free plan: limited)
4. **Download PDF:** Click "Download PDF" button when ready
5. **Spell check:** Built-in spell checker (click "Aa" icon)

---

## Next Steps After Upload

1. ✅ Upload files to Overleaf
2. ✅ Verify PDF compiles correctly
3. ✅ Create Draw.io figures (see FIGURE_SPECIFICATIONS.md)
4. ✅ Upload figures to Overleaf
5. ✅ Final proofreading
6. ✅ Download PDF for submission

---

**Estimated Time:** 10-15 minutes to upload and verify

**Recommended:** Use AAAI 2026 template for latest formatting requirements
