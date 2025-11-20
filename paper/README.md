# AAAI Conference Paper

**Title:** Reinforcement Learning-Enhanced Model Predictive Control with Sequential Transfer Learning for Multi-UAV Systems

## Files Included

```
paper/
├── main.tex                  # Main paper LaTeX source
├── references.bib            # Bibliography file
├── README.md                 # This file
├── compile.bat               # Windows compilation script
├── compile.sh                # Linux/Mac compilation script
├── figures/                  # Figures directory (add your figures here)
└── tables/                   # Tables directory
```

## Compilation Instructions

### Option 1: Automated Compilation (Recommended)

**Windows:**
```bash
compile.bat
```

**Linux/Mac:**
```bash
chmod +x compile.sh
./compile.sh
```

### Option 2: Manual Compilation

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option 3: Overleaf

1. Create new project on Overleaf
2. Upload all files from `paper/` directory
3. Set compiler to `pdfLaTeX`
4. Click "Recompile"

## Required LaTeX Packages

The paper uses the following packages (usually included in standard TeX distributions):

- aaai24 (AAAI 2024 style file)
- times, helvet, courier (fonts)
- graphicx (figures)
- amsmath, amssymb (math symbols)
- algorithm, algorithmic (algorithms)
- booktabs (tables)
- subcaption (subfigures)
- multirow (table formatting)

## Adding Figures

Place your figure files (PDF, PNG, or JPG) in the `figures/` directory and reference them in the LaTeX:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.8\columnwidth]{figures/your_figure.pdf}
\caption{Your caption here}
\label{fig:your_label}
\end{figure}
```

## AAAI Submission Guidelines

### Page Limit
- **Main paper:** 7 pages (including references)
- **Appendix:** Additional 1 page allowed

### Formatting Requirements
- ✅ Anonymous submission (author names hidden)
- ✅ Letterpaper size
- ✅ Two-column format
- ✅ Times font
- ✅ Proper citations

### Checklist Before Submission

- [ ] Remove author names and affiliations
- [ ] Remove acknowledgments mentioning funding or institutions
- [ ] Check all citations are formatted correctly
- [ ] Verify figures are high quality (300 DPI minimum)
- [ ] Ensure page limit is met (7 pages + 1 appendix)
- [ ] Run spell check
- [ ] Verify all references are accessible
- [ ] Include supplementary code/data links if applicable

## Paper Structure

1. **Abstract** (200 words) - Done ✓
2. **Introduction** - Done ✓
3. **Related Work** - Done ✓
4. **Problem Formulation** - Done ✓
5. **Methodology** - Done ✓
6. **Experimental Setup** - Done ✓
7. **Results** - Done ✓
8. **Discussion** - Done ✓
9. **Conclusion** - Done ✓

## Key Contributions Highlighted

1. Automated MPC hyperparameter optimization via RL
2. Sequential transfer learning across heterogeneous UAVs
3. Production-ready system with checkpoint resilience
4. Comprehensive experimental validation (4 platforms, 3 algorithms)

## Results Summary

- **Tracking accuracy:** < 5cm for lightweight UAVs
- **Training time reduction:** 75-80% via transfer learning
- **Mass range:** 200× variation (0.027kg to 5.5kg)
- **Transfer efficiency:** 20% fine-tuning steps achieve comparable performance

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{anonymous2024rl,
  title={Reinforcement Learning-Enhanced Model Predictive Control with Sequential Transfer Learning for Multi-UAV Systems},
  author={Anonymous},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

## Contact

For questions regarding the paper or implementation, please contact the authors through the AAAI submission system.

---

**Last Updated:** November 2024
**Conference:** AAAI 2024
**Track:** Robotics and Autonomous Systems
