# SETUP AND USAGE GUIDE
## How to Use the Project Tracking System

**IMPORTANT:** The tracking files are **NOT automatically updated**. You (Dr. Khan) must update them manually after each work session. This guide shows you exactly how.

---

## 📁 STEP 1: INITIAL SETUP (Do This Once)

### 1.1 Create Your Project Directory

```bash
# Create the main project directory
mkdir -p ~/rl_mpc_drones
cd ~/rl_mpc_drones

# Create the full directory structure
mkdir -p docs/{phase_01_simulator,phase_02_pid,phase_03_obstacles,phase_04_mpc,phase_05_multi_platform,phase_06_rl,phase_07_transfer}
mkdir -p configs
mkdir -p src/{simulators,controllers,mpc,rl_agents,utils}
mkdir -p tests
mkdir -p results
mkdir -p checkpoints
mkdir -p models

# Verify structure was created
tree -L 2
```

### 1.2 Initialize Git Repository

```bash
# Initialize git
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Results and models (too large for git)
results/videos/*.mp4
models/*.zip

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Temporary files
*.tmp
*.log
EOF

# First commit
git add .gitignore
git commit -m "Initial commit: Project structure and gitignore"
```

### 1.3 Place Tracking Files

**Now copy all 6 tracking files you downloaded into the project root:**

```bash
# Assuming you downloaded the files to ~/Downloads/
# Adjust path as needed

cd ~/rl_mpc_drones

cp ~/Downloads/NEW_SESSION_BRIEFING.md .
cp ~/Downloads/PROJECT_STATE_TRACKER.md .
cp ~/Downloads/PROGRESS_LOG.md .
cp ~/Downloads/SESSION_STARTER_CHECKLIST.md .
cp ~/Downloads/TRACKING_SYSTEM_GUIDE.md .
cp ~/Downloads/PROJECT_AT_A_GLANCE.md .

# Verify they're there
ls -la *.md

# Should see:
# NEW_SESSION_BRIEFING.md
# PROJECT_STATE_TRACKER.md
# PROGRESS_LOG.md
# SESSION_STARTER_CHECKLIST.md
# TRACKING_SYSTEM_GUIDE.md
# PROJECT_AT_A_GLANCE.md
```

### 1.4 Create Initial Project Files

```bash
# Create README.md
cat > README.md << 'EOF'
# RL-Enhanced MPC for Heterogeneous Drone Fleets

## Overview
This project implements an RL-enhanced Model Predictive Control system that outperforms expert manual tuning for heterogeneous drone fleets.

## Key Results (Target)
- 62-81% time savings vs expert tuning
- 11-47% better tracking performance
- 1.34±0.01m RMSE across 200× mass variation
- Transfer learning reduces training by 75%

## Quick Start
1. Read `NEW_SESSION_BRIEFING.md` for project overview
2. Read `PROJECT_STATE_TRACKER.md` for current status
3. Follow `QUICK_START_GUIDE.md` for development roadmap

## Documentation
- `NEW_SESSION_BRIEFING.md` - Start here every session
- `PROJECT_STATE_TRACKER.md` - Current project status
- `PROGRESS_LOG.md` - Detailed work log
- `docs/` - Phase-specific documentation

## Author
Dr. Abdul Manan Khan

## Status
Project Start: November 2025
Current Phase: Phase 0 - Setup
EOF

# Create requirements.txt stub
cat > requirements.txt << 'EOF'
# Core Dependencies
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0

# Simulation (choose one)
# Option 1: PyBullet (recommended)
gym-pybullet-drones>=1.0.0
pybullet>=3.2.0

# Option 2: Webots (alternative)
# webots>=2023.1.0

# MPC
casadi>=3.6.0

# RL
stable-baselines3[extra]>=2.0.0
gymnasium>=0.29.0

# Utilities
pyyaml>=6.0
tensorboard>=2.13.0
# wandb>=0.15.0  # Optional: cloud logging

# Development
pytest>=7.0.0
black>=23.0.0
flake8>=6.0.0
EOF

# Commit everything
git add .
git commit -m "Setup: Add tracking files, README, and requirements"
```

### 1.5 Verify Setup Complete

```bash
# Check your directory structure
cd ~/rl_mpc_drones
ls -la

# Should see:
# .git/
# .gitignore
# README.md
# requirements.txt
# NEW_SESSION_BRIEFING.md
# PROJECT_STATE_TRACKER.md
# PROGRESS_LOG.md
# SESSION_STARTER_CHECKLIST.md
# TRACKING_SYSTEM_GUIDE.md
# PROJECT_AT_A_GLANCE.md
# docs/
# configs/
# src/
# tests/
# results/
# checkpoints/
# models/

echo "✅ Setup complete!"
```

---

## 🔄 STEP 2: DAILY WORKFLOW (Do This Every Session)

### 2.1 START OF SESSION (5 minutes)

**Every time you open your PC and start working:**

```bash
# 1. Navigate to project
cd ~/rl_mpc_drones

# 2. Check git status
git status
git log --oneline -5

# 3. Open your text editor with tracking files
# Example with VS Code:
code NEW_SESSION_BRIEFING.md PROJECT_STATE_TRACKER.md PROGRESS_LOG.md

# Or any text editor you prefer:
# vim NEW_SESSION_BRIEFING.md
# nano NEW_SESSION_BRIEFING.md
# gedit NEW_SESSION_BRIEFING.md
```

**Then READ these files in order:**

```
Step 1: Read NEW_SESSION_BRIEFING.md (60 seconds)
   ↓
   Understand: What are we building? What's the current status?
   
Step 2: Read PROJECT_STATE_TRACKER.md (2-3 minutes)
   ↓
   Focus on these sections:
   - CURRENT PROJECT STATUS (which phase, progress %)
   - Current phase section (read the phase you're in)
   - IMMEDIATE NEXT STEPS (what to do today)
   
Step 3: Read PROGRESS_LOG.md (2 minutes)
   ↓
   Read the LAST session entry only
   - What did you do last time?
   - What decisions were made?
   - What should you do next?
```

**Now you're ready to work! You should know:**
- ✅ What phase you're in
- ✅ What you did last session
- ✅ What to do today
- ✅ Any blockers

### 2.2 DURING WORK SESSION

**While you're working:**

```bash
# Keep a text file or notepad open for quick notes
# Example: work-notes.txt

# Write down as you work:
- What you're implementing
- Decisions you make
- Problems you encounter
- Solutions you find
- Test results

# Example notes:
# "10:30 - Started implementing MPC dynamics"
# "11:15 - Decided to use CasADi Opti interface instead of nlp"
# "12:00 - Problem: IPOPT not converging, trying smaller time step"
# "12:30 - Solution: Reduced horizon from 20 to 10, now works"
# "13:00 - Test results: Hover RMSE = 0.09m (target < 0.1m) ✓"
```

**Reference these when needed:**
- `PROJECT_STATE_TRACKER.md` - Check current phase details
- `SESSION_STARTER_CHECKLIST.md` - Quick commands
- `QUICK_START_GUIDE.md` - Phase instructions

### 2.3 END OF SESSION (15 minutes - CRITICAL!)

**⚠️ THIS IS THE MOST IMPORTANT PART - Don't skip this!**

#### Step A: Update PROGRESS_LOG.md (10 minutes)

```bash
# Open PROGRESS_LOG.md
code PROGRESS_LOG.md  # or your preferred editor

# Scroll to the bottom
# Copy the session template (found at bottom of PROGRESS_LOG.md)
# Fill in all sections using your work notes

# Example filled session:
```

```markdown
### Session #2 - November 21, 2025

**Date:** 2025-11-21
**Time:** 09:00 - 13:00
**Duration:** 4 hours
**Current Phase:** Phase 1 - Simulator Selection
**Phase Progress:** 80%

#### What I Did Today
1. Researched PyBullet vs Webots
2. Installed PyBullet and gym-pybullet-drones
3. Tested basic drone spawn and control
4. Verified all 12 states are accessible
5. Measured real-time factor: 2.1x

#### Decisions Made
- **Decision 1:** Chose PyBullet over Webots
  - Rationale: Faster, better RL integration, simpler API
  - Alternative considered: Webots (better visuals but slower)
  - Why this way: Speed is more important than visuals for training

#### Challenges Faced
1. IPOPT installation failed with pip
2. gym-pybullet-drones had import errors

#### Solutions Found
1. Used conda for IPOPT: `conda install -c conda-forge ipopt`
2. Installed gym-pybullet-drones from source instead of pip

#### Code Changes
- Files created: src/simulators/pybullet_interface.py
- Files created: tests/test_simulator_capabilities.py
- Lines of code: ~150

#### Test Results
- Tests run: 5
- Tests passing: 5/5 ✓
- Key findings: Real-time factor 2.1x exceeds target of 1.5x

#### Performance Metrics
- Real-time factor: 2.1x (target: >1.5x) ✓
- States accessible: 12/12 ✓

#### Lessons Learned
1. PyBullet installation is straightforward on Linux
2. Always use conda for scientific packages when pip fails
3. Real-time factor depends heavily on GPU

#### Tomorrow's Goals
1. Complete Phase 1 documentation
2. Create phase_01_checkpoint.yaml
3. Begin Phase 2: PID controller research

#### Blockers
None

#### Questions/Concerns
None

#### Time Breakdown
- Research: 60 min
- Installation: 45 min
- Testing: 90 min
- Documentation: 45 min
- Total: 4 hours

#### Mood/Confidence
😊 Confident - Phase 1 nearly complete, ahead of schedule!

**Overall Assessment:** Excellent progress, Phase 1 almost done
```

**Then SAVE the file!**

#### Step B: Update PROJECT_STATE_TRACKER.md (5 minutes)

```bash
# Open PROJECT_STATE_TRACKER.md
code PROJECT_STATE_TRACKER.md

# Update these sections:
```

**1. Update "CURRENT PROJECT STATUS":**
```markdown
### Overall Progress: 12% Complete  # Changed from 0%

```
Phase 1: Simulator        [████████░░] 80%  # Changed from 0%
Phase 2: PID              [░░░░░░░░░░] 0%
...
```

### Current Phase: PHASE 1 - Simulator Selection  # Update if changed
**Status:** Nearly Complete  # Changed from "Not Started"
**Started:** November 21, 2025  # Add start date
**Target Completion:** November 22, 2025  # Add target
```

**2. Update "DAILY PROGRESS LOG" section:**
```markdown
### Session #2 - November 21, 2025
**Duration:** 4 hours
**Phase:** Phase 1 - Simulator Selection
**Accomplishments:**
- Chose PyBullet simulator
- Completed installation and testing
- Verified all requirements met
- 80% complete

**Decisions Made:**
- PyBullet over Webots (speed priority)

**Next Session Goals:**
1. Complete Phase 1 documentation
2. Create checkpoint
3. Begin Phase 2
```

**3. Update "IMMEDIATE NEXT STEPS":**
```markdown
### Must Do Now (Session #3)
1. Complete Phase 1 documentation
2. Create phase_01_checkpoint.yaml
3. Git commit and tag phase-1-complete

### Must Do Next (Session #4)
1. Research PID controller implementation
2. Design PID gains using Bryson's Rule
3. Create configs/pid_default.yaml
```

**4. Update Phase 1 section:**
```markdown
### Phase 1: Simulator Selection (2-3 days)
**Status:** 80% COMPLETE  # Changed
**Dependencies:** None

**Exit Criteria:**
- [x] Simulator installed and running  # Mark complete
- [x] Can spawn drone and control it  # Mark complete
- [x] Can read all 12 states  # Mark complete
- [x] Real-time factor > 1.5×  # Mark complete
- [ ] Documentation created  # Still need to do
- [ ] Checkpoint saved  # Still need to do

**Blockers:** None
**Notes:** PyBullet working excellently, real-time factor 2.1x
```

**Then SAVE the file!**

#### Step C: Git Commit (2 minutes)

```bash
# Stage all changes
git add .

# Commit with descriptive message
git commit -m "Session 2: Phase 1 80% complete - PyBullet setup and testing"

# If phase complete, add tag
# git tag phase-1-complete
# (Do this when Phase 1 is 100% done)

# View recent commits
git log --oneline -3
```

#### Step D: Create Checkpoint (if phase complete)

**Only do this when a phase is 100% complete:**

```bash
# Create checkpoint file
code checkpoints/phase_01_checkpoint.yaml

# Fill in with this template:
```

```yaml
phase: 1
phase_name: "Simulator Selection"
status: "Complete"
completion_date: "2025-11-21"

metrics:
  simulator_chosen: "PyBullet"
  real_time_factor: 2.1
  states_accessible: 12
  installation_time_minutes: 45

decisions:
  - decision: "Chose PyBullet over Webots"
    rationale: "Faster, simpler API, better for RL training"
  - decision: "Using gym-pybullet-drones package"
    rationale: "Pre-built drone models save development time"

challenges:
  - challenge: "IPOPT installation failed with pip"
    solution: "Used conda install instead"
  - challenge: "gym-pybullet-drones import errors"
    solution: "Installed from source"

lessons_learned:
  - "PyBullet installation is straightforward on Linux"
  - "Real-time factor >2x achievable with good GPU"
  - "Always prefer conda for scientific packages"

time_spent_hours: 4
time_target_hours: 16-24
ahead_behind: "Ahead of schedule"

files_created:
  - "src/simulators/pybullet_interface.py"
  - "tests/test_simulator_capabilities.py"
  - "docs/phase_01_simulator/SIMULATOR_REPORT.md"

tests_passing: 5
tests_total: 5

next_phase: 2
next_phase_name: "PID Controller"
next_phase_ready: true

notes: |
  Phase went smoothly. PyBullet is working perfectly.
  Real-time factor exceeds requirements.
  Ready to proceed to PID controller implementation.
```

**Save and commit:**
```bash
git add checkpoints/phase_01_checkpoint.yaml
git commit -m "Phase 1 Complete: Add checkpoint"
git tag phase-1-complete
```

---

## 🔄 STEP 3: WHEN YOU RESTART YOUR PC

**Every time you shut down and restart:**

```bash
# 1. Open terminal and navigate to project
cd ~/rl_mpc_drones

# 2. Check git status (see what was done)
git status
git log --oneline -5

# 3. Open Claude and share these files
# Upload to Claude:
# - NEW_SESSION_BRIEFING.md
# - PROJECT_STATE_TRACKER.md
# - PROGRESS_LOG.md (last 2-3 entries)

# 4. Ask Claude:
"I'm resuming my RL-MPC project. Here are my tracking files. 
Please read them and tell me:
1. What phase am I in?
2. What did I do last session?
3. What should I do next?
4. Are there any blockers?"

# Claude will read the files and give you full context!
```

---

## 🤖 STEP 4: HOW CLAUDE USES THE TRACKING FILES

### When You Start a New Claude Session:

**You say:**
```
"Hi Claude, I'm resuming my RL-Enhanced MPC drone project. 
I've uploaded my tracking files. Please review them and 
brief me on where we are."
```

**Claude will:**
1. Read `NEW_SESSION_BRIEFING.md` first (60 sec)
2. Read `PROJECT_STATE_TRACKER.md` (understand current state)
3. Read last entry in `PROGRESS_LOG.md` (understand recent work)
4. Tell you:
   - Current phase and progress %
   - What you accomplished last session
   - What you should work on next
   - Any blockers or concerns

**Claude's response will be like:**
```
"Welcome back, Dr. Khan! Based on your tracking files:

📍 Current Status:
- Phase: Phase 1 (Simulator Selection) - 80% complete
- Overall project: 12% complete
- Started: November 21, 2025

✅ Last Session (Session #2 - Nov 21):
- Chose PyBullet simulator
- Completed installation and testing
- Achieved 2.1x real-time factor (exceeds 1.5x target)
- All 12 states accessible
- 5/5 tests passing

🎯 What's Next:
1. Complete Phase 1 documentation
2. Create phase_01_checkpoint.yaml
3. Git commit and tag 'phase-1-complete'
4. Then begin Phase 2: PID Controller

🚧 Blockers: None

You're ahead of schedule! Ready to finish Phase 1?"
```

**Then you can start working, and Claude will remember everything!**

---

## 📋 STEP 5: MAINTENANCE CHECKLIST

### Daily (After Each Session):
- [ ] Updated PROGRESS_LOG.md with session entry
- [ ] Updated PROJECT_STATE_TRACKER.md status sections
- [ ] Git commit with clear message
- [ ] Know tomorrow's top 3 priorities

### Weekly:
- [ ] Created weekly summary in PROGRESS_LOG.md
- [ ] Reviewed overall progress vs timeline
- [ ] Checked if tracking system is working well

### After Each Phase:
- [ ] Created checkpoint YAML file
- [ ] Created phase completion summary in PROGRESS_LOG.md
- [ ] Git tagged: `phase-X-complete`
- [ ] Updated PROJECT_STATE_TRACKER.md phase status to "Complete"

---

## ⚠️ CRITICAL REMINDERS

### 🔴 TRACKING FILES ARE NOT AUTOMATIC!

**You must manually update them.** The tracking files do NOT update themselves.

**After each session, you MUST:**
1. ✍️ Write session entry in PROGRESS_LOG.md
2. ✍️ Update status in PROJECT_STATE_TRACKER.md
3. 💾 Git commit everything

**If you skip this, Claude won't know what happened!**

### 🔴 WHERE TO SAVE TRACKING FILES

**Save them in your project ROOT directory:**
```
~/rl_mpc_drones/
├── NEW_SESSION_BRIEFING.md         ← Here
├── PROJECT_STATE_TRACKER.md        ← Here
├── PROGRESS_LOG.md                 ← Here
├── SESSION_STARTER_CHECKLIST.md    ← Here
├── TRACKING_SYSTEM_GUIDE.md        ← Here
├── PROJECT_AT_A_GLANCE.md          ← Here
├── README.md
├── requirements.txt
├── docs/
├── src/
└── ...
```

**NOT in subdirectories!** Keep them at the top level.

### 🔴 UPLOADING TO CLAUDE

When you start a new Claude session:

**Upload these 3 files (in order):**
1. NEW_SESSION_BRIEFING.md
2. PROJECT_STATE_TRACKER.md
3. PROGRESS_LOG.md

**Optional (if needed):**
4. Current phase documentation from docs/
5. Recent checkpoint file from checkpoints/

**Don't upload all files every time** - just these key tracking files.

---

## 🎯 EXAMPLE: COMPLETE WORKFLOW

### Day 1 - First Work Session

```bash
# Morning - Start session
cd ~/rl_mpc_drones
git status

# Read tracking files (5 min)
cat NEW_SESSION_BRIEFING.md | less
cat PROJECT_STATE_TRACKER.md | less
tail -50 PROGRESS_LOG.md

# Work for 4 hours
# Take notes as you work

# Evening - End session (15 min)
code PROGRESS_LOG.md
# Add session entry, fill all sections

code PROJECT_STATE_TRACKER.md  
# Update: Current Status, Daily Progress Log, Immediate Next Steps, Phase status

git add .
git commit -m "Session 1: Phase 1 started - PyBullet research"

# Shut down PC
```

### Day 2 - Resume After Restart

```bash
# Morning - PC restarted
cd ~/rl_mpc_drones
git status
git log --oneline -3

# Open Claude, upload files:
# - NEW_SESSION_BRIEFING.md
# - PROJECT_STATE_TRACKER.md  
# - PROGRESS_LOG.md

# Ask Claude:
"Hi Claude, I'm resuming my project. Please review my 
tracking files and brief me on where we are."

# Claude responds with full context
# You start working immediately

# Work for 4 hours

# Evening - End session (15 min)
# Update PROGRESS_LOG.md
# Update PROJECT_STATE_TRACKER.md
git commit -m "Session 2: Phase 1 80% - PyBullet setup complete"

# Shut down PC
```

### Day 3 - Complete Phase 1

```bash
# Morning - Resume
cd ~/rl_mpc_drones

# Upload to Claude and get briefing
# Work for 2 hours

# Complete Phase 1!

# Create checkpoint
code checkpoints/phase_01_checkpoint.yaml
# Fill in all sections

# Update tracking files
code PROGRESS_LOG.md  # Add session, mark phase complete
code PROJECT_STATE_TRACKER.md  # Mark Phase 1 100% complete

# Commit and tag
git add .
git commit -m "Phase 1 Complete: Simulator selection done"
git tag phase-1-complete

# Celebrate! 🎉
```

---

## 🆘 TROUBLESHOOTING

### Problem: "I forgot to update tracking files last session"

**Solution:**
1. Check git log: `git log --oneline -5` to see what you committed
2. Reconstruct what you did from git commits
3. Update PROGRESS_LOG.md now (better late than never)
4. Update PROJECT_STATE_TRACKER.md to current state
5. Add note: "Reconstructed from git history"

### Problem: "Claude doesn't understand where we are"

**Solution:**
1. Check: Did you update tracking files last session?
2. Upload these files to Claude:
   - NEW_SESSION_BRIEFING.md
   - PROJECT_STATE_TRACKER.md (with updates)
   - PROGRESS_LOG.md (with last entries)
3. If still confused, manually tell Claude:
   "We're in Phase X, we've completed Y, we need to do Z"

### Problem: "Tracking files are out of sync"

**Solution:**
1. Trust git log as the source of truth
2. Check what was actually committed
3. Update PROJECT_STATE_TRACKER.md to match reality
4. Add reconciliation note in PROGRESS_LOG.md

### Problem: "I don't know what to update"

**Solution:**
1. Use the templates in PROGRESS_LOG.md
2. Copy session template, fill in all sections
3. At minimum update:
   - What you did
   - Current phase progress %
   - Next steps
4. Check SESSION_STARTER_CHECKLIST.md for guidance

---

## ✅ FINAL VERIFICATION

**Setup is complete when:**
- [ ] Project directory created: `~/rl_mpc_drones/`
- [ ] All 6 tracking files in root directory
- [ ] Git initialized and first commit made
- [ ] README.md and requirements.txt created
- [ ] You understand the daily workflow
- [ ] You know how to update tracking files
- [ ] You know how to use Claude with tracking files

**Ready to start development!** 🚀

---

## 📞 QUICK REFERENCE CARD

**Print this and keep visible:**

```
┌─────────────────────────────────────────────────┐
│         DAILY WORKFLOW QUICK REFERENCE          │
└─────────────────────────────────────────────────┘

START SESSION (5 min):
1. cd ~/rl_mpc_drones
2. git status && git log --oneline -5
3. Read: NEW_SESSION_BRIEFING.md
4. Read: PROJECT_STATE_TRACKER.md (status section)
5. Read: PROGRESS_LOG.md (last entry)
6. Start working!

END SESSION (15 min):
1. Update PROGRESS_LOG.md (add session entry)
2. Update PROJECT_STATE_TRACKER.md (update status)
3. git add . && git commit -m "Session X: ..."
4. If phase done: Create checkpoint + git tag

RESTART PC:
1. cd ~/rl_mpc_drones
2. Upload tracking files to Claude
3. Ask: "Brief me on where we are"
4. Continue working

CRITICAL: Update tracking files EVERY session!
```

---

**Last Updated:** November 20, 2025
**Version:** 1.0
