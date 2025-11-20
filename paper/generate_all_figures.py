"""
Master script to generate all figures for AAAI paper
Generates professional, publication-quality figures
"""

import os
import sys

# Ensure figures directory exists
os.makedirs('D:/rl_tuned_mpc/paper/figures', exist_ok=True)

print("=" * 60)
print("Generating Professional Figures for AAAI Paper")
print("=" * 60)
print()

# Generate Figure 2: System Architecture
print("[*] Generating Figure 2: System Architecture...")
exec(open('D:/rl_tuned_mpc/paper/generate_figure_2_architecture.py', encoding='utf-8').read())
print()

# Generate Figure 3: Transfer Learning Flow
print("[*] Generating Figure 3: Transfer Learning Flow...")
exec(open('D:/rl_tuned_mpc/paper/generate_figure_3_transfer.py', encoding='utf-8').read())
print()

# Generate Figure 4: PyBullet Environment
print("[*] Generating Figure 4: PyBullet Environment...")
exec(open('D:/rl_tuned_mpc/paper/generate_figure_4_pybullet.py', encoding='utf-8').read())
print()

print("=" * 60)
print("[SUCCESS] All 4 figures generated successfully!")
print("=" * 60)
print()
print("Generated files:")
print("  1. system_architecture.pdf (Figure 2)")
print("  2. system_architecture.png (preview)")
print("  3. transfer_learning_flow.pdf (Figure 3)")
print("  4. transfer_learning_flow.png (preview)")
print("  5. pybullet_environment.pdf (Figure 4)")
print("  6. pybullet_environment.png (preview)")
print()
print("Plus existing:")
print("  - training_results.png (Figure 1)")
print()
print("Total: 4 figures for the paper")
print()
print("Next steps:")
print("  1. All figures already integrated in main.tex")
print("  2. Upload all files to Overleaf")
print("  3. Compile and verify page count")
print()
