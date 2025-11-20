# How to Install LaTeX on Windows

**⚠️ WARNING: This takes 1-2 hours. Use Overleaf instead for instant results.**

## Why This Guide Exists

You're seeing this error:
```
'pdflatex' is not recognized as an internal or external command
```

This means LaTeX is not installed on your computer.

## Recommended: Don't Install - Use Overleaf Instead

**Seriously, use Overleaf:**
- No installation needed
- Works in 5 minutes
- See: QUICK_START_OVERLEAF.md

**Still want to install locally? Continue below...**

---

## Installing MiKTeX (LaTeX for Windows)

### Step 1: Download MiKTeX

1. Go to: https://miktex.org/download
2. Click "Download" button
3. Save file: `basic-miktex-x64.exe` (~280 MB)

### Step 2: Run Installer

1. Double-click downloaded file
2. Click "Next"
3. Choose installation type:
   - **Recommended:** "Install MiKTeX for all users"
   - Alternative: "Install just for me"
4. Click "Next"
5. Choose installation directory:
   - Default: `C:\Program Files\MiKTeX`
   - Or choose another location
6. Click "Next"
7. Settings:
   - Package installation: **"Yes"** (auto-install missing packages)
   - Paper size: **A4** or **Letter**
8. Click "Next"
9. Click "Start"
10. **Wait 30-60 minutes** (installation is large)

### Step 3: Verify Installation

1. **Close ALL terminal windows**
2. **Open NEW Command Prompt:**
   - Press `Win + R`
   - Type: `cmd`
   - Press Enter
3. **Test pdflatex:**
   ```cmd
   pdflatex --version
   ```
4. **Should see:**
   ```
   MiKTeX-pdfTeX 4.x (MiKTeX x.x)
   ```

If you see this, LaTeX is installed! ✅

### Step 4: First Compilation

1. **Navigate to paper directory:**
   ```cmd
   cd D:\rl_tuned_mpc\paper
   ```

2. **Run compile script:**
   ```cmd
   compile.bat
   ```

3. **First run will install packages:**
   - MiKTeX will prompt to install missing packages
   - Click "Install" for each package
   - **This takes 15-30 minutes**
   - Be patient!

4. **Common packages that will be installed:**
   - aaai style files
   - graphics packages
   - algorithm packages
   - font packages
   - Many more...

### Step 5: Check Output

If successful, you'll see:
```
========================================
  Compilation Complete!
========================================

Output file: main.pdf
```

The PDF will be in: `D:\rl_tuned_mpc\paper\main.pdf`

---

## Troubleshooting

### Issue: "pdflatex still not recognized"

**Cause:** PATH not updated
**Fix:**
1. Restart computer
2. Open new terminal
3. Try again

### Issue: Package installation fails

**Cause:** Network or repository issue
**Fix:**
1. Open MiKTeX Console (Start Menu → MiKTeX Console)
2. Click "Updates" tab
3. Click "Check for updates"
4. Install updates
5. Try compile.bat again

### Issue: "File aaai24.sty not found"

**Cause:** AAAI style file not in MiKTeX repository
**Fix:**
1. Download from: https://aaai.org/authorkit24-2/
2. Extract `aaai24.sty`
3. Copy to: `D:\rl_tuned_mpc\paper\`
4. Try compile.bat again

### Issue: Compilation very slow

**This is normal!**
- First compilation: 5-10 minutes
- Subsequent: 1-2 minutes
- MPC generates large aux files

---

## Alternative: Install TeX Live (More Complete)

If MiKTeX doesn't work, try TeX Live:

1. **Download:** https://www.tug.org/texlive/acquire-netinstall.html
2. **File:** install-tl-windows.exe
3. **Run installer**
4. **Choose:** "Install TeX Live"
5. **Wait:** 1-2 hours (installs EVERYTHING)
6. **Disk space needed:** 7+ GB

---

## After Installation

### Disk Space Used

- MiKTeX Basic: ~1 GB
- After packages: ~3 GB
- TeX Live Full: ~7 GB

### Keeping Updated

MiKTeX Console:
1. Open: Start Menu → MiKTeX Console
2. Click "Updates"
3. Install updates monthly

---

## Still Not Working?

### Use Overleaf (Seriously!)

**Benefits:**
- Zero installation time
- Zero disk space
- Always up to date
- Works on any computer
- Built-in collaboration
- Automatic backups

**See:** QUICK_START_OVERLEAF.md

---

## Summary

**Time Investment:**
- Download: 5 minutes
- Install: 30-60 minutes
- First compilation: 15-30 minutes
- **Total: 1-2 hours**

**Compare to Overleaf:**
- Setup: 0 minutes
- Upload files: 5 minutes
- Compile: 30 seconds
- **Total: 5 minutes**

**Recommendation:** Use Overleaf unless you have a specific reason to compile locally.

---

**Need help?** See QUICK_START_OVERLEAF.md for the easier solution!
