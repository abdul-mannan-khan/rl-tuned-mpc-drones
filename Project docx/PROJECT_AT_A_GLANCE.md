# PROJECT AT-A-GLANCE
## One-Page Reference Card

**Keep this visible while working!**

---

## 🎯 PROJECT IDENTITY

**RL-Enhanced MPC for Heterogeneous Drone Fleets**

**Goal:** Beat expert manual tuning with automated RL system

**Key Numbers:**
- 62-81% time savings | 11-47% better tracking | 200× mass range

---

## 📊 CURRENT STATUS

```
┌──────────────────────────────────────────────────────┐
│  Phase: _____   Progress: ____%   Status: _____     │
│  Last Session: _______   Next: _______              │
│  Blocker: _______                                   │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 THE 7 PHASES

```
□ 1. Simulator      (2-3d)  → PyBullet/Webots
□ 2. PID            (2-3d)  → Basic controller
□ 3. Obstacles      (3-4d)  → Navigate (optional)
□ 4. MPC ⭐         (5-7d)  → Baseline to beat
□ 5. Multi-Platform (3-4d)  → 4 drones manual
□ 6. RL ⭐          (5-7d)  → Beat baseline
□ 7. Transfer       (6-8d)  → Transfer to 3 more
```

---

## 🎯 TARGET METRICS

| Platform | Manual | RL Target | Status |
|----------|--------|-----------|--------|
| Crazyflie| ~1.5m  | 1.34m     | □      |
| Racing   | ~1.5m  | 1.34m     | □      |
| Generic  | ~1.7m  | 1.34m     | □      |
| Heavy    | ~2.0m  | 1.34m     | □      |

**Time:** 6.1h total (vs 16-32h manual)

---

## 📁 ESSENTIAL FILES

**Read Every Session:**
1. NEW_SESSION_BRIEFING.md (60s)
2. PROJECT_STATE_TRACKER.md (3m)
3. PROGRESS_LOG.md last entry (2m)

**Update Every Session:**
1. PROGRESS_LOG.md (add entry)
2. PROJECT_STATE_TRACKER.md (update status)
3. Git commit

---

## ⚡ QUICK COMMANDS

```bash
# Status check
pwd && git status

# Recent work
git log --oneline -5
tail -20 PROGRESS_LOG.md

# Environment check
python -c "import pybullet, casadi; from stable_baselines3 import PPO"

# What's next?
grep "IMMEDIATE NEXT STEPS" PROJECT_STATE_TRACKER.md -A 5
```

---

## ✅ SESSION CHECKLIST

**Before Work:**
- [ ] Read briefing + tracker
- [ ] Know current phase
- [ ] Know next 3 priorities
- [ ] No unresolved blockers

**After Work:**
- [ ] Update progress log
- [ ] Update tracker status
- [ ] Git commit
- [ ] Note tomorrow's goals

---

## 🔑 KEY CONCEPTS

**Bryson's Rule:** Q = 1/(error)²  
**4 Platforms:** 0.027kg → 5.5kg  
**Training:** 20k steps base, 5k transfer  
**Target:** 1.34±0.01m RMSE all platforms

---

## 🚨 CRITICAL RULES

1. Never skip exit criteria
2. Update tracking every session
3. MPC must be perfect first
4. Always use Bryson's Rule
5. Commit to git frequently

---

## 🆘 HELP DECISION

```
Stuck 30min? → Check docs
Stuck 1hr?   → Read phase detail
Stuck 2hr?   → Review checkpoints
Stuck 4hr?   → Try different approach
```

---

## 📞 CURRENT PRIORITIES

**Top 3 for today:**
1. ______________________________
2. ______________________________
3. ______________________________

**Next session:**
1. ______________________________
2. ______________________________
3. ______________________________

---

## 💾 GIT WORKFLOW

```bash
# Start of day
git pull
git status

# During work (frequently)
git add .
git commit -m "Phase X: Clear description"

# End of phase
git tag phase-X-complete
git push
```

---

## 📊 PROGRESS TRACKING

```
Phase 1: [░░░░░░░░░░] 0%
Phase 2: [░░░░░░░░░░] 0%
Phase 3: [░░░░░░░░░░] 0%
Phase 4: [░░░░░░░░░░] 0% ⭐
Phase 5: [░░░░░░░░░░] 0%
Phase 6: [░░░░░░░░░░] 0% ⭐
Phase 7: [░░░░░░░░░░] 0%

Overall: [░░░░░░░░░░] 0%
```

---

## 🎓 TODAY'S PHASE

**Phase:** _______  
**Goal:** _______  
**Exit Criteria:**
- [ ] _______
- [ ] _______
- [ ] _______

**Doc:** docs/phase_XX/README.md

---

## ⏱️ TIME TRACKING

**Session Start:** _______  
**Planned Duration:** _______  
**Actual Duration:** _______  
**Focused Work:** _______  
**Debugging:** _______  
**Documentation:** _______

---

## 🔧 ENVIRONMENT STATUS

- [ ] Correct directory
- [ ] Git status clean
- [ ] Dependencies installed
- [ ] Tests passing
- [ ] Documentation current

---

## 📝 NOTES SPACE

**Today's Key Insight:**
_________________________________

**Tomorrow's Reminder:**
_________________________________

**Don't Forget:**
_________________________________

---

## 🎯 SUCCESS CRITERIA

**Project Complete When:**
- [ ] All 7 phases done
- [ ] RL beats manual (time + RMSE)
- [ ] 1.34±0.01m all platforms
- [ ] All docs complete
- [ ] Code reproducible

---

## 💪 MOTIVATION

**You're building:**
- System that beats experts
- 16-32 hour time savings
- Publication-worthy research
- Democratized control

**Keep going!** 🚀

---

**Last Updated:** _______  
**Next Update:** End of today's session

---

## 🖨️ PRINT THIS!

**Keep it visible while working for:**
- Quick status reference
- Command reminders
- Checklist verification
- Motivation boost

