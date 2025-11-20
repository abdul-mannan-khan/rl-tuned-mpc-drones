# TRACKING SYSTEM GUIDE
## How Everything Fits Together

---

## 📊 The Complete Tracking System

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│              YOUR PROJECT TRACKING SYSTEM               │
│                                                         │
└─────────────────────────────────────────────────────────┘
                            │
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌───────────────────┐                 ┌───────────────────┐
│                   │                 │                   │
│  START OF SESSION │                 │   END OF SESSION  │
│                   │                 │                   │
└───────────────────┘                 └───────────────────┘
        │                                       │
        │                                       │
        ▼                                       ▼
┌───────────────────┐                 ┌───────────────────┐
│ 1. READ THIS:     │                 │ 1. UPDATE THESE:  │
│                   │                 │                   │
│ NEW_SESSION_      │                 │ PROGRESS_LOG.md   │
│ BRIEFING.md       │                 │ (add session)     │
│ (60 sec)          │                 │                   │
│                   │                 │ PROJECT_STATE_    │
│                   │                 │ TRACKER.md        │
│                   │                 │ (update status)   │
└───────────────────┘                 │                   │
        │                             │                   │
        ▼                             └───────────────────┘
┌───────────────────┐                         │
│ 2. READ THIS:     │                         ▼
│                   │                 ┌───────────────────┐
│ PROJECT_STATE_    │                 │ 2. GIT COMMIT:    │
│ TRACKER.md        │                 │                   │
│ - Current Status  │                 │ git add .         │
│ - Last Progress   │                 │ git commit -m ""  │
│ - Next Steps      │                 │                   │
│ (2-3 min)         │                 └───────────────────┘
│                   │                         │
└───────────────────┘                         ▼
        │                             ┌───────────────────┐
        ▼                             │ 3. CHECKPOINT     │
┌───────────────────┐                 │    (if phase      │
│ 3. START WORKING! │                 │     complete):    │
│                   │                 │                   │
│ Use SESSION_      │                 │ Create phase_XX_  │
│ STARTER_          │                 │ checkpoint.yaml   │
│ CHECKLIST.md      │                 │                   │
│ during work       │                 └───────────────────┘
└───────────────────┘
```

---

## 📁 File Purposes (When to Use Each)

### 🚀 NEW_SESSION_BRIEFING.md
**READ FIRST - Every session start**

**Purpose:** Ultra-fast context recovery  
**When:** Beginning of EVERY session  
**Time:** 60 seconds  
**Contains:**
- Project essentials
- Current status snapshot
- Where to find everything
- Quick commands

**Use Case:** "I just opened my computer, what was I doing?"

---

### 📊 PROJECT_STATE_TRACKER.md
**MAIN STATUS - Update regularly**

**Purpose:** Single source of truth for project state  
**When:**
- Read at session start (after briefing)
- Update at session end
- Update when phase changes
- Check when confused

**Time:** 
- Read: 2-3 minutes
- Update: 5 minutes

**Contains:**
- Detailed phase status
- All metrics and targets
- Blockers and risks
- Immediate next steps

**Use Case:** "What exactly is the status of Phase 4?"

---

### 📝 PROGRESS_LOG.md
**DETAILED DIARY - Add entry every session**

**Purpose:** Complete historical record  
**When:**
- Write at end of EVERY session
- Read last entry at session start
- Review for weekly summaries
- Reference when writing papers

**Time:**
- Write: 10-15 minutes per session
- Read: 2-3 minutes

**Contains:**
- What you did each session
- Decisions made
- Challenges and solutions
- Time breakdowns
- Lessons learned

**Use Case:** "What did I try last time when I had this problem?"

---

### ✅ SESSION_STARTER_CHECKLIST.md
**WORKFLOW GUIDE - Use during work**

**Purpose:** Ensure you don't forget anything  
**When:**
- Quick reference during session
- Before starting work (pre-check)
- After finishing work (post-check)

**Time:** 2 minutes (scan through)

**Contains:**
- Pre-work checklist
- Post-work checklist
- Common commands
- Phase-specific quick starts

**Use Case:** "Did I forget to update anything?"

---

## 🔄 The Workflow Loop

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│                  SESSION WORKFLOW                   │
│                                                     │
└─────────────────────────────────────────────────────┘

1. ARRIVE (5 min)
   ├─ Read: NEW_SESSION_BRIEFING.md (60 sec)
   ├─ Read: PROJECT_STATE_TRACKER.md (2 min)
   │  └─ Current Status
   │  └─ Immediate Next Steps
   └─ Read: PROGRESS_LOG.md last entry (2 min)
   
2. WORK (Variable)
   ├─ Follow current phase documentation
   ├─ Reference SESSION_STARTER_CHECKLIST.md
   ├─ Write code / Run tests / Debug
   └─ Make progress on current phase
   
3. DEPART (15 min)
   ├─ Update: PROGRESS_LOG.md (10 min)
   │  └─ Add complete session entry
   ├─ Update: PROJECT_STATE_TRACKER.md (3 min)
   │  ├─ Current phase progress
   │  ├─ Immediate next steps
   │  └─ Any blockers
   ├─ Git commit (2 min)
   │  └─ Meaningful commit message
   └─ Checkpoint (if phase complete)
      └─ Create checkpoints/phase_XX_checkpoint.yaml

4. REPEAT
   └─ Return to step 1 next session
```

---

## 🎯 Decision Tree: Which File Do I Need?

```
START HERE
    │
    ▼
┌─────────────────────┐
│ What do you need?   │
└─────────────────────┘
    │
    ├─ "Just started, need context" 
    │   → NEW_SESSION_BRIEFING.md
    │
    ├─ "What's the current status of Phase X?"
    │   → PROJECT_STATE_TRACKER.md → Phase X section
    │
    ├─ "What did I do yesterday?"
    │   → PROGRESS_LOG.md → Last entry
    │
    ├─ "What should I do next?"
    │   → PROJECT_STATE_TRACKER.md → Immediate Next Steps
    │
    ├─ "Did I forget to update something?"
    │   → SESSION_STARTER_CHECKLIST.md → Post-work section
    │
    ├─ "How do I do Phase X?"
    │   → QUICK_START_GUIDE.md → Phase X
    │   → Or docs/phase_XX/README.md
    │
    ├─ "What are the research questions?"
    │   → RESEARCH_FOCUS_SUMMARY.md
    │
    ├─ "I'm confused about everything"
    │   → NEW_SESSION_BRIEFING.md (start over)
    │
    └─ "How does tracking work?"
        → TRACKING_SYSTEM_GUIDE.md (this file!)
```

---

## 📚 Reading Order (New to Project)

If someone completely new (including a new Claude) needs to understand the project:

**Tier 1 - Essential (15 min)**
1. NEW_SESSION_BRIEFING.md (60 sec)
2. PROJECT_STATE_TRACKER.md 
   - Sections: Project Identity, Current Status, Current Phase
   - Time: 5 min
3. PROGRESS_LOG.md
   - Read last 2-3 entries
   - Time: 5 min
4. QUICK_START_GUIDE.md
   - Skim phases
   - Time: 5 min

**Tier 2 - Detailed (30 min)**
5. RESEARCH_FOCUS_SUMMARY.md (10 min)
6. Current phase documentation in docs/ (10 min)
7. DEVELOPMENT_ROADMAP_DETAILED.md (10 min)

**Tier 3 - Reference (as needed)**
8. SESSION_STARTER_CHECKLIST.md
9. Phase-specific detailed docs
10. Original project requirements

---

## 🔑 Key Principles

### 1. Single Source of Truth
**PROJECT_STATE_TRACKER.md** is the authoritative source.  
All other files support or feed into it.

### 2. Update Frequency
- **Every Session:** PROGRESS_LOG.md (add entry)
- **Every Session:** PROJECT_STATE_TRACKER.md (update sections)
- **Every Phase:** Create checkpoint file
- **Weekly:** Review and summarize

### 3. Redundancy is Good
Some information appears in multiple places:
- Current phase: Briefing, Tracker, Progress Log
- Next steps: Briefing, Tracker

**Why?** Different contexts need different details

### 4. Write for Future You
Assume you'll forget everything in 2 weeks.  
Write documentation that explains clearly.

### 5. Commit Often
Every meaningful change should be committed.  
Documentation changes are meaningful changes!

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Not updating after session
**Problem:** Next session starts confused  
**Solution:** Set 15-min timer at session end for updates

### ❌ Mistake 2: Vague progress entries
**Bad:** "Worked on MPC"  
**Good:** "Implemented MPC dynamics, tested hover, RMSE=0.08m"

### ❌ Mistake 3: Not committing tracking files
**Problem:** Lose progress if computer crashes  
**Solution:** Commit everything, including docs

### ❌ Mistake 4: Skipping NEW_SESSION_BRIEFING.md
**Problem:** Waste time remembering context  
**Solution:** Always read it first (only 60 seconds!)

### ❌ Mistake 5: Not creating checkpoints
**Problem:** Can't easily resume after breaks  
**Solution:** Create checkpoint.yaml after each phase

---

## 📋 Checkpoint System Explained

### What is a Checkpoint?

A checkpoint is a YAML file created after completing each phase that captures:
- What was accomplished
- Final metrics achieved
- Key decisions made
- Lessons learned
- How to resume if needed

### Checkpoint Template

```yaml
# checkpoints/phase_01_checkpoint.yaml

phase: 1
phase_name: "Simulator Selection"
status: "Complete"
completion_date: "2025-11-XX"

metrics:
  simulator_chosen: "PyBullet"
  real_time_factor: 2.1
  states_accessible: 12
  installation_time_minutes: 45

decisions:
  - decision: "Chose PyBullet over Webots"
    rationale: "Faster, simpler API, better for RL training"
  - decision: "Using gym-pybullet-drones package"
    rationale: "Pre-built drone models and environments"

challenges:
  - challenge: "IPOPT installation issues"
    solution: "Used conda install instead of pip"

lessons_learned:
  - "PyBullet installation is straightforward on Linux"
  - "Real-time factor >2x is achievable with good GPU"

time_spent_hours: 3.5
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
  Phase went smoothly. PyBullet is working well.
  Ready to proceed to PID controller implementation.
```

### When to Create Checkpoints
- After completing each phase
- Before taking a break of >3 days
- After any major milestone

### How to Use Checkpoints
- Resume project: Read latest checkpoint first
- Compare phases: Look at time_spent vs time_target
- Learn from past: Review lessons_learned sections

---

## 🎓 Best Practices Summary

### Daily Routine

**Morning (5 min):**
1. Read NEW_SESSION_BRIEFING.md
2. Check PROJECT_STATE_TRACKER.md status
3. Review last PROGRESS_LOG.md entry
4. Know your top 3 goals for today

**During Work:**
- Keep PROJECT_STATE_TRACKER.md open
- Reference SESSION_STARTER_CHECKLIST.md
- Make notes as you work (for progress log)

**Evening (15 min):**
1. Update PROGRESS_LOG.md with session entry
2. Update PROJECT_STATE_TRACKER.md status
3. Git commit all changes
4. Note tomorrow's priorities

### Weekly Routine

**Weekly Review (30 min):**
1. Create weekly summary in PROGRESS_LOG.md
2. Update overall progress % in tracker
3. Review metrics vs targets
4. Adjust plans if needed

### Phase Completion Routine

**When Phase Complete (30 min):**
1. Create checkpoint YAML file
2. Create phase completion summary in PROGRESS_LOG.md
3. Update phase status in PROJECT_STATE_TRACKER.md
4. Git tag: `git tag phase-X-complete`
5. Review lessons learned
6. Prep for next phase

---

## 🚀 Quick Start for New Claude Sessions

```
NEW CLAUDE SESSION DETECTED
    │
    ▼
READ THIS FILE ORDER:
    │
    ├─ 1. NEW_SESSION_BRIEFING.md (60 sec)
    │   └─ Get basic context
    │
    ├─ 2. PROJECT_STATE_TRACKER.md (3 min)
    │   ├─ PROJECT IDENTITY section
    │   ├─ CURRENT PROJECT STATUS section  
    │   └─ Current phase section details
    │
    ├─ 3. PROGRESS_LOG.md (2 min)
    │   └─ Last 1-2 session entries
    │
    └─ 4. START WORKING!
        └─ Check PROJECT_STATE_TRACKER.md
            → IMMEDIATE NEXT STEPS
            → Begin working on top priority

TOTAL TIME TO CONTEXT: < 10 minutes
TOTAL TIME TO PRODUCTIVITY: < 15 minutes
```

---

## 📊 Success Metrics for Tracking System

**Good tracking system achieves:**

✅ New session setup: < 10 minutes  
✅ Context recovery: 100% (no lost information)  
✅ Resume after 1 week break: < 15 minutes  
✅ Resume after 1 month break: < 30 minutes  
✅ New person understands project: < 2 hours  
✅ Find specific decision: < 5 minutes  
✅ Track overall progress: Always current  
✅ Documentation overhead: < 15 min/session  

**If not achieving these, system needs adjustment!**

---

## 🔧 System Maintenance

### Monthly Review
- [ ] Check all tracking files are up to date
- [ ] Verify git commits are descriptive
- [ ] Ensure checkpoints exist for completed phases
- [ ] Review if tracking system is working well
- [ ] Adjust templates if needed

### Red Flags
- Progress log hasn't been updated in 2+ sessions
- Git commits are vague ("update", "changes", etc.)
- Checkpoint files missing
- Can't remember what you did yesterday
- Taking >15 min to understand where you left off

**If you see red flags → Fix immediately!**

---

## 💡 Pro Tips

1. **Use git aliases for common operations**
   ```bash
   git config alias.quick "commit -am"
   git config alias.latest "log --oneline -5"
   ```

2. **Keep tracking files always visible**
   - Use split-screen with tracker on one side
   - Code on other side

3. **Write progress entries AS you work**
   - Keep PROGRESS_LOG.md open
   - Add notes in real-time
   - Format at session end

4. **Use markdown preview**
   - Most IDEs can preview .md files
   - Makes documentation easier to read

5. **Automate what you can**
   ```bash
   # Quick status check
   alias project-status="cat PROJECT_STATE_TRACKER.md | grep 'Current Phase' && git status"
   ```

---

## 📞 Troubleshooting the Tracking System

### Problem: "I forgot to update last session"
**Solution:**
1. Check git log to see what was changed
2. Reconstruct from git commits
3. Update now (better late than never)

### Problem: "Tracking files are out of sync"
**Solution:**
1. Use git history to find truth
2. Update PROJECT_STATE_TRACKER.md as master
3. Add reconciliation note in PROGRESS_LOG.md

### Problem: "Too much documentation overhead"
**Solution:**
1. Use templates (copy & fill)
2. Write notes during work, format at end
3. Focus on key decisions, not every detail

### Problem: "Can't find information"
**Solution:**
1. Use file search: `grep -r "keyword" *.md`
2. Check git log: `git log --grep="keyword"`
3. Review checkpoint files
4. Improve documentation going forward

---

## ✅ Final Checklist

### Am I using the tracking system correctly?

- [ ] I read NEW_SESSION_BRIEFING.md at start of EVERY session
- [ ] I update PROGRESS_LOG.md at end of EVERY session
- [ ] I update PROJECT_STATE_TRACKER.md regularly
- [ ] I create checkpoints after each phase
- [ ] I commit to git after meaningful changes
- [ ] I can resume work after a break in <15 minutes
- [ ] A new person could understand the project from my docs
- [ ] I know exactly what to work on next
- [ ] I can find past decisions quickly

**If all checked → You're using the system perfectly! 🎉**

---

**Remember: The tracking system exists to HELP you, not slow you down!**

**If it feels like a burden, you're probably over-documenting. Focus on:**
1. Current status (what phase, what's done)
2. Next actions (what to do next)
3. Key decisions (why did you choose X over Y)
4. Blockers (what's preventing progress)

**Everything else is optional!**

---

**Last Updated:** November 20, 2025  
**Maintenance:** Update if tracking system changes
