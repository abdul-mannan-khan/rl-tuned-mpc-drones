# NEW SESSION BRIEFING
## 60-Second Context Recovery

**READ THIS FIRST when starting any new session!**

---

## ⚡ THE ESSENTIALS (30 seconds)

### Project Name
**RL-Enhanced MPC for Heterogeneous Drone Fleets**

### What We're Building
An automated MPC tuning system using RL that **beats human experts** in both time and performance.

### The Big Numbers (Memorize These)
- **62-81% faster** than expert manual tuning
- **11-47% better tracking** (1.34m vs 1.5-2.2m RMSE)
- **200× mass variation** (0.027kg to 5.5kg drones)
- **75% training reduction** via transfer learning

### Current Status (Check PROJECT_STATE_TRACKER.md for updates)
```
Overall Progress: 0%
Current Phase: Phase 0 - Project Setup
Last Session: #1 - November 20, 2025
Next Action: Create project directory structure
```

---

## 📍 WHERE TO FIND EVERYTHING (20 seconds)

### Must-Read Files (in order)
1. **PROJECT_STATE_TRACKER.md** ← Your bible (read sections 1-3)
2. **PROGRESS_LOG.md** ← Last session's work (read last entry)
3. **SESSION_STARTER_CHECKLIST.md** ← Pre/post work checklist

### Reference When Needed
- **QUICK_START_GUIDE.md** - Phase overviews and timelines
- **RESEARCH_FOCUS_SUMMARY.md** - Research questions and value prop
- **DEVELOPMENT_ROADMAP_DETAILED.md** - Detailed how-tos
- **docs/phase_XX/** - Current phase documentation

---

## 🎯 YOUR IMMEDIATE ACTIONS (10 seconds)

### Before You Start Working
1. [ ] Read PROJECT_STATE_TRACKER.md → "CURRENT PROJECT STATUS"
2. [ ] Read PROGRESS_LOG.md → Last entry
3. [ ] Understand current phase goals
4. [ ] Check for blockers

### After You Finish Working
1. [ ] Update PROGRESS_LOG.md with session entry
2. [ ] Update PROJECT_STATE_TRACKER.md status sections
3. [ ] Git commit with clear message
4. [ ] Note tomorrow's top 3 priorities

---

## 🔑 KEY CONTEXT

### The 7 Development Phases
```
1. Simulator (2-3d)     → Get PyBullet/Webots running
2. PID (2-3d)           → Basic flight controller
3. Obstacles (3-4d)     → Navigate obstacles (optional)
4. MPC (5-7d) ⭐        → Nonlinear MPC baseline
5. Multi-Platform (3-4d) → Test on 4 drones (manual tuning)
6. RL (5-7d) ⭐         → Beat manual baseline with PPO
7. Transfer (6-8d)      → Transfer to 3 other platforms

Total: 27-38 days (5-8 weeks)
```

### The 4 Drone Platforms
- **Crazyflie:** 0.027kg (train RL here first)
- **Racing:** 0.800kg
- **Generic:** 2.500kg
- **Heavy-Lift:** 5.500kg (hardest to tune)

### Critical Success Criteria
- [ ] RL achieves 1.34±0.01m RMSE across all 4 platforms
- [ ] RL training takes ~6.1 hours total (vs 16-32h manual)
- [ ] Transfer learning works with 5,000 steps (vs 20,000 from scratch)
- [ ] Beats expert manual tuning in time AND performance

---

## 🚨 CRITICAL RULES

1. **Never skip phase exit criteria** - One broken phase breaks everything after
2. **Update tracking files after EVERY session** - Future you depends on it
3. **MPC must be perfect before RL** - Phase 4 is the foundation
4. **Always use Bryson's Rule** - Don't start from random weights
5. **Commit to git frequently** - Your safety net

---

## 💡 CURRENT PHASE QUICK REF

### Phase 0: Project Setup (Current)
**Goal:** Create project structure and initialize tracking  
**Duration:** Few hours  
**Exit Criteria:**
- [ ] Directory structure created
- [ ] Git initialized
- [ ] Basic docs created (README, requirements.txt)
- [ ] All tracking files ready

**Next Phase:** Phase 1 - Simulator Selection

---

## 🆘 IF YOU'RE CONFUSED

### Lost? (Read in order)
1. This file (you're here) ✓
2. PROJECT_STATE_TRACKER.md → "PROJECT IDENTITY"
3. PROJECT_STATE_TRACKER.md → "CURRENT PROJECT STATUS"
4. PROGRESS_LOG.md → Last 2 entries
5. Current phase section in tracker

### Stuck? (Try these)
1. Check docs/TROUBLESHOOTING.md (when it exists)
2. Review current phase checkpoint (when it exists)
3. Check git log to see what was done: `git log --oneline -10`
4. Read phase documentation in detail

### Very Stuck? (After 2+ hours)
1. Check if you're in the right phase
2. Verify all previous phases are actually complete
3. Review phase exit criteria
4. Consider if you need to go back and fix something

---

## 📊 SUCCESS METRICS SNAPSHOT

### Performance Targets
| Platform | Manual Baseline | RL Target | Status |
|----------|----------------|-----------|--------|
| Crazyflie | ~1.5m | 1.33-1.35m | Not tested |
| Racing | ~1.3-1.8m | 1.34±0.01m | Not tested |
| Generic | ~1.4-1.9m | 1.34±0.01m | Not tested |
| Heavy-Lift | ~1.6-2.2m | 1.34±0.01m | Not tested |

### Time Targets
| Task | Manual | RL+Transfer | Savings |
|------|--------|-------------|---------|
| 4 Platforms | 16-32h | 6.1h | 62-81% |
| Single Platform | 4-8h | 3.3h (first) | ~50% |
| Transfer (each) | 4-8h | 1h | 75-88% |

---

## 🎓 CORE CONCEPTS CHEAT SHEET

**Bryson's Rule:**  
`Q = 1 / (max_acceptable_error)²`  
Used for intelligent MPC weight initialization

**Sequential Transfer:**  
Train: Crazyflie (20k steps)  
Transfer: Racing (5k steps) → Generic (5k) → Heavy (5k)  
Knowledge builds progressively

**PPO Hyperparameters:**  
- Learning rate: 3e-4
- Batch size: 64
- n_steps: 2048
- Parallel envs: 4
- Training steps: 20,000

**Target RMSE:**  
1.34±0.01m across ALL platforms (this is what makes the research compelling)

---

## 📁 FILE STRUCTURE OVERVIEW

```
rl_mpc_drones/
├── PROJECT_STATE_TRACKER.md    ← Single source of truth
├── PROGRESS_LOG.md              ← Detailed session logs
├── SESSION_STARTER_CHECKLIST.md ← Pre/post work checklist
├── NEW_SESSION_BRIEFING.md      ← This file
├── README.md                    ← Project description
├── requirements.txt             ← Python dependencies
├── .gitignore                   ← Git ignore rules
│
├── docs/                        ← Phase documentation
│   ├── phase_01_simulator/
│   ├── phase_02_pid/
│   └── ...
│
├── configs/                     ← Configuration files
│   ├── pid_default.yaml
│   ├── mpc_crazyflie.yaml
│   └── rl_ppo.yaml
│
├── src/                         ← Source code
│   ├── simulators/
│   ├── controllers/
│   ├── mpc/
│   ├── rl_agents/
│   └── utils/
│
├── tests/                       ← Test suite
├── results/                     ← Experimental results
├── checkpoints/                 ← Phase checkpoints
└── models/                      ← Trained RL models
```

---

## ⚡ QUICK COMMANDS

```bash
# Check current status
cat PROJECT_STATE_TRACKER.md | grep "Current Phase"
tail -20 PROGRESS_LOG.md

# See what was done recently
git log --oneline -5

# Check what tests exist
ls tests/

# See latest results
ls -lt results/ | head -5

# Quick environment check (when installed)
python -c "import pybullet; import casadi; from stable_baselines3 import PPO; print('All OK')"
```

---

## 🎯 YOUR 3-STEP START PROCESS

### Step 1: Load Context (2 min)
```bash
# Read these sections in order:
1. This file (done!)
2. PROJECT_STATE_TRACKER.md → "CURRENT PROJECT STATUS"
3. PROGRESS_LOG.md → Last entry
```

### Step 2: Verify Environment (1 min)
```bash
# Check you're in right place
pwd  # Should be in rl_mpc_drones/
git status  # Check for uncommitted changes
```

### Step 3: Start Working (immediately)
- Look at PROJECT_STATE_TRACKER.md → "IMMEDIATE NEXT STEPS"
- Pick the top priority item
- Start working on it
- Update tracking files when done

---

## 💪 MOTIVATION REMINDER

**Why This Matters:**
- Democratizes advanced control (no PhD needed)
- Saves 16-32 hours per deployment
- Better performance than human experts
- Scales to unlimited platforms
- Publication-worthy research

**You're Building Something Important!**

This isn't just a coding project - it's a system that:
1. Outperforms human experts
2. Saves massive amounts of time
3. Makes advanced control accessible
4. Provides publishable research
5. Has real-world impact

**Keep going! The research community needs this!**

---

## ✅ PRE-WORK VERIFICATION

Before starting work, verify:
- [ ] I know which phase I'm in
- [ ] I know what was done last session  
- [ ] I know what to do next
- [ ] I have the right documentation open
- [ ] I'm in the correct directory
- [ ] Git status is clean (or I know why it's not)

**If all checked → START WORKING!**

---

## 📞 FINAL CHECKLIST

### Information Hierarchy (Use in Order)
1. **NEW_SESSION_BRIEFING.md** (this file) - Start here every time
2. **PROJECT_STATE_TRACKER.md** - Detailed current status
3. **PROGRESS_LOG.md** - Historical context
4. **SESSION_STARTER_CHECKLIST.md** - Pre/post work routine
5. **QUICK_START_GUIDE.md** - Phase overviews
6. **Phase-specific docs** - Detailed how-tos

### Git Hygiene
- Commit after every meaningful change
- Write clear commit messages
- Tag phase completions
- Push to remote regularly (if using remote)

### Documentation Hygiene  
- Update PROGRESS_LOG.md after every session
- Update PROJECT_STATE_TRACKER.md status sections
- Create checkpoint files after each phase
- Keep documentation current

---

## 🚀 READY TO GO!

**You now have enough context to start working!**

**Next Steps:**
1. Open PROJECT_STATE_TRACKER.md in another window
2. Check "IMMEDIATE NEXT STEPS" section
3. Start working on top priority
4. Update tracking files when done

**Time to First Productivity:** < 5 minutes from reading this file

**Good luck! You've got this!** 💪

---

**Last Updated:** November 20, 2025  
**Version:** 1.0  
**Maintenance:** Update this file if project structure changes significantly

