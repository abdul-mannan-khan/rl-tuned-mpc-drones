# How to Get AAAI Style File

The paper requires `aaai24.sty` to compile. You have two options:

## Option 1: Download AAAI Author Kit (If you want to compile locally)

1. Visit: https://aaai.org/authorkit24-2/
2. Download the AAAI-24 Author Kit ZIP file
3. Extract `aaai24.sty` from the ZIP
4. Copy `aaai24.sty` to this `paper/` directory
5. Run `compile.bat` (Windows) or `./compile.sh` (Linux/Mac)

## Option 2: Use Overleaf (Recommended - No download needed)

Overleaf has AAAI templates built-in:

1. Go to https://www.overleaf.com/
2. Create account (free)
3. New Project → Upload Project
4. Zip all files in this `paper/` directory
5. Upload the ZIP file
6. Click "Recompile" - PDF is generated automatically!

**Overleaf advantages:**
- No LaTeX installation required
- AAAI style file included automatically
- Real-time PDF preview
- Collaboration features
- Version control

## Option 3: For AAAI 2025 or Later

If submitting to AAAI 2025 or newer:
- Check for updated author kit at: https://aaai.org/authorkit/
- Download the appropriate year's style file
- Update the `\usepackage{aaai24}` line in `main.tex` to match

---

**Recommended approach:** Use Overleaf - it's the easiest and most reliable method.
