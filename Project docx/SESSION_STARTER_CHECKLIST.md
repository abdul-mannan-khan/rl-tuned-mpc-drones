# SESSION STARTER CHECKLIST
## Quick Context Recovery for RL-Enhanced MPC Project

**Purpose:** Get any Claude session up to speed in under 3 minutes

---

## ⚡ 3-MINUTE CONTEXT RECOVERY

### Step 1: Project Identity (30 seconds)
**What are we building?**
RL-enhanced MPC system that beats expert manual tuning for drone fleets

**Core numbers:**
- 62-81% time savings
- 11-47% better tracking (1.34m vs 1.5-2.2m RMSE)
- 200× mass variation (0.027kg to 5.5kg)

### Step 2: Where Are We? (30 seconds)
**Check:** `PROJECT_STATE_TRACKER.md` → "CURRENT PROJECT STATUS"

**Quick Status:**
```
Current Phase: _____________
Progress: ____%
Last Session: _____________
```

### Step 3: What Did We Last Do? (60 seconds)
**Check:** `PROJECT_STATE_TRACKER.md` → "DAILY PROGRESS LOG" (last entry)

**Last Accomplishments:**
- 
- 
- 

**Last Decisions:**
- 
- 

### Step 4: What's Next? (60 seconds)
**Check:** `PROJECT_STATE_TRACKER.md` → "IMMEDIATE NEXT STEPS"

**Top 3 priorities:**
1. 
2. 
3. 

---

## 📋 PRE-WORK CHECKLIST

Before starting work, verify:

### Environment Check
- [ ] Correct directory: `pwd` shows `rl_mpc_drones/`
- [ ] Git status clean: `git status`
- [ ] Virtual environment active (if using one)
- [ ] Previous session committed: `git log -1`

### Context Check
- [ ] Read PROJECT_STATE_TRACKER.md sections 1-3
- [ ] Understand current phase goals
- [ ] Know what was done last session
- [ ] Identify any blockers

### Ready to Work When:
- [ ] Know exactly what to work on
- [ ] Have needed documentation open
- [ ] Understand success criteria
- [ ] No unresolved blockers

---

## 🔄 POST-WORK CHECKLIST

After each work session, update:

### Update PROJECT_STATE_TRACKER.md
- [ ] Add entry to "DAILY PROGRESS LOG"
- [ ] Update current phase status
- [ ] Update "IMMEDIATE NEXT STEPS"
- [ ] Note any blockers or decisions

### Git Hygiene
- [ ] Stage changes: `git add .`
- [ ] Commit with clear message: `git commit -m "Phase X: [what you did]"`
- [ ] Optional: Tag if phase complete: `git tag phase-X-complete`

### Create Checkpoint (if phase complete)
- [ ] Create `checkpoints/phase_XX_checkpoint.yaml`
- [ ] Document key metrics
- [ ] Note lessons learned
- [ ] Update overall progress percentage

---

## 📁 KEY FILES QUICK REFERENCE

**Always Open These:**
1. `PROJECT_STATE_TRACKER.md` - Current status
2. `PROGRESS_LOG.md` - Detailed daily log
3. Current phase doc: `docs/phase_XX/README.md`

**Reference When Needed:**
- `QUICK_START_GUIDE.md` - Phase overviews
- `RESEARCH_FOCUS_SUMMARY.md` - Research questions
- `DEVELOPMENT_ROADMAP_DETAILED.md` - Detailed instructions

**Check When Stuck:**
- `docs/TROUBLESHOOTING.md` - Common issues
- `checkpoints/phase_XX_checkpoint.yaml` - What worked before

---

## 🚀 PHASE-SPECIFIC QUICK STARTS

### If in Phase 1 (Simulator)
**Goal:** Get simulator running  
**Test:** `python tests/test_simulator_capabilities.py`  
**Exit:** All 12 states readable, real-time factor > 1.5×

### If in Phase 2 (PID)
**Goal:** Stable hover control  
**Test:** `python tests/test_pid_controller.py`  
**Exit:** Hover RMSE < 0.1m

### If in Phase 3 (Obstacles)
**Goal:** Navigate obstacle course  
**Test:** `python tests/test_obstacle_avoidance.py`  
**Exit:** 100% success on 3-obstacle course

### If in Phase 4 (MPC) ⭐
**Goal:** Nonlinear MPC working  
**Test:** `python tests/test_mpc_controller.py`  
**Exit:** Circular tracking RMSE < 2.0m, solve time < 50ms  
**Critical:** This is the baseline RL must beat!

### If in Phase 5 (Multi-Platform)
**Goal:** MPC on all 4 drones  
**Test:** `python tests/test_all_platforms.py`  
**Exit:** Manual tuning RMSE documented for all 4

### If in Phase 6 (RL) ⭐
**Goal:** PPO beats manual baseline  
**Test:** `python src/rl/train_ppo.py`  
**Monitor:** TensorBoard at `http://localhost:6006`  
**Exit:** RMSE ≤ 1.35m after 20,000 steps

### If in Phase 7A (Transfer)
**Goal:** Transfer to 3 platforms  
**Test:** `python src/rl/transfer_learning.py --target racing`  
**Exit:** 1.34±0.01m RMSE with 5,000 steps each

### If in Phase 7B (Algorithms)
**Goal:** Compare 5 RL algorithms  
**Test:** `python src/rl/train_[algorithm].py`  
**Exit:** Performance comparison table complete

---

## 🎯 ONE-LINE PROJECT STATUS

Fill this out for instant context:

```
Phase [__]/7 | [___]% Complete | Last: [_______] | Next: [_______] | Blocker: [_______]
```

Example:
```
Phase 4/7 | 50% Complete | Last: MPC dynamics | Next: CasADi setup | Blocker: None
```

---

## 💡 CONTEXT RECOVERY STRATEGIES

### If Totally Lost (10 min recovery)
1. Open `PROJECT_STATE_TRACKER.md`
2. Read "PROJECT IDENTITY" section
3. Read "CURRENT PROJECT STATUS" section
4. Read last 3 entries in "DAILY PROGRESS LOG"
5. Check `git log --oneline -10`
6. Read current phase section in detail
7. You're now caught up!

### If Partially Lost (5 min recovery)
1. Check current phase in tracker
2. Read last log entry
3. Check "IMMEDIATE NEXT STEPS"
4. Continue working

### If Just Forgot Details (2 min recovery)
1. Check tracker's current phase section
2. Review exit criteria
3. Continue working

---

## 🔧 COMMON RECOVERY COMMANDS

```bash
# Where am I?
pwd
git status
git log --oneline -5

# What was I doing?
tail -20 PROJECT_STATE_TRACKER.md | head -10
cat checkpoints/phase_*_checkpoint.yaml | tail -1

# What tests exist?
ls tests/
ls -lt tests/ | head -5  # Most recent tests

# What's the latest result?
ls -lt results/ | head -5

# What models do I have?
find models/ -name "*.zip" 2>/dev/null

# Quick environment check
python -c "import pybullet; print('PyBullet OK')"
python -c "import casadi; print('CasADi OK')"
python -c "from stable_baselines3 import PPO; print('SB3 OK')"
```

---

## 📊 PROGRESS VISUALIZATION

### Phase Completion Matrix
```
1. Simulator      [▱▱▱▱▱▱▱▱▱▱] 0%   | Target: 2-3 days
2. PID            [▱▱▱▱▱▱▱▱▱▱] 0%   | Target: 2-3 days
3. Obstacles      [▱▱▱▱▱▱▱▱▱▱] 0%   | Target: 3-4 days
4. MPC ⭐         [▱▱▱▱▱▱▱▱▱▱] 0%   | Target: 5-7 days
5. Multi-Platform [▱▱▱▱▱▱▱▱▱▱] 0%   | Target: 3-4 days
6. RL ⭐          [▱▱▱▱▱▱▱▱▱▱] 0%   | Target: 5-7 days
7. Transfer       [▱▱▱▱▱▱▱▱▱▱] 0%   | Target: 6-8 days

Overall: [▱▱▱▱▱▱▱▱▱▱] 0% (0/7 phases)
```

Update this after each phase!

---

## 🎓 KEY CONCEPTS REFRESHER

**Bryson's Rule:** Q = 1/(max_error)²  
Used for intelligent MPC weight initialization

**Transfer Learning:** Train on Crazyflie → fine-tune on others  
Reduces training from 20k to 5k steps (75% reduction)

**Target Performance:** 1.34±0.01m RMSE  
This beats manual tuning: 1.5-2.2m

**Sequential Transfer:** Crazyflie → Racing → Generic → Heavy  
Knowledge builds progressively

**4 Platforms:**
- Crazyflie: 0.027kg (lightest)
- Racing: 0.800kg
- Generic: 2.500kg
- Heavy-Lift: 5.500kg (hardest to tune)

**200× Mass Variation:** 5.500/0.027 = 203.7×

---

## 🚨 RED FLAGS

Watch for these issues:

### Code Red (Stop Everything)
- MPC solver never converges
- Simulator crashes repeatedly
- Git repository corrupted
- Can't reproduce previous results

### Code Yellow (Investigate Soon)
- Tests starting to fail
- Performance degrading
- Training not converging
- Documentation falling behind

### Code Green (All Good)
- Tests passing
- Checkpoints up to date
- Documentation current
- Progress on schedule

---

## 📞 HELP DECISION TREE

```
Can't figure it out in 30 min?
    ↓
Check troubleshooting guide
    ↓
Still stuck after 1 hour?
    ↓
Check phase documentation
    ↓
Still stuck after 2 hours?
    ↓
Review checkpoints and logs
    ↓
Still stuck after 4 hours?
    ↓
Consider asking for help or trying different approach
```

**Remember:** 
- Check docs first
- Check logs second  
- Try different approach third
- Ask for help fourth

---

## ✅ SESSION SUCCESS CRITERIA

A good work session includes:

- [ ] Made measurable progress
- [ ] Updated PROJECT_STATE_TRACKER.md
- [ ] Committed changes to git
- [ ] Tests still passing (or new tests added)
- [ ] No new blockers introduced
- [ ] Know exactly what to do next session

**If all checked:** Session successful! 🎉

---

## 🎯 ULTIMATE QUICK START

**Absolute fastest context recovery (90 seconds):**

1. Open `PROJECT_STATE_TRACKER.md`
2. Read "CURRENT PROJECT STATUS" (15 sec)
3. Read last "DAILY PROGRESS LOG" entry (30 sec)
4. Read "IMMEDIATE NEXT STEPS" (15 sec)
5. Skim current phase "Exit Criteria" (30 sec)
6. **START WORKING!**

---

**This checklist is your friend. Use it every session!**

**Last Updated:** November 20, 2025  
**Next Update:** After current session
