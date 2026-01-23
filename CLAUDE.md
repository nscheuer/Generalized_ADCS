# Project Notes for Claude

## Python Environment

Use the project virtual environment for running Python:
```bash
/home/pmckeen/Generalized_ADCS/venv/bin/python
```

## CLAUDE.md Maintenance

- **Sync all CLAUDE.md files**: When asked to update a CLAUDE.md file with general instructions, also update all other CLAUDE.md files (e.g., `~/CLAUDE.md` and project-specific ones)
- **Ask if unsure**: If it's unclear whether an instruction is general or project-specific, ask the user before propagating changes

---

## Ongoing Work Sessions

### Paper Interview for Sharpening (Session: "paper interview for sharpening")
**Last Updated**: 2026-01-23
**Status**: Active - Session 3 complete, papers updated with lit review & framework

**Purpose**: Dissertation defense prep - sharpening 4 papers + dissertation through structured interview

**Key Files**:
- Interview notes: `papers/mckeen_dissertation_interview_session.md`
- Papers now in repo: `papers/` folder
  - `papers/3MTQ+1RW/main2.tex` (3+1 paper)
  - `papers/MTQ_Planner/main2.tex` (Planner paper)
  - `papers/Package_Paper/main2.tex` (Package paper)
  - `papers/Generalized_ACS/main2.tex` (Generalized Control paper)
- Original Windows locations: `/mnt/c/Users/LV - Patrick McKeen/Writing/`

**Session 3 Accomplishments** (2026-01-23):
- Added literature review TODOs to all 4 papers
- Added bolt-on framework structure to Generalized Control paper:
  - Stage 1: Goal Modification (full↔reduced attitude conversion)
  - Stage 2: Existing Control Law (user's choice - unchanged)
  - Stage 3: Compensation (disturbance, gyroscopic, desaturation)
  - Stage 4: Allocation & Bounds (LP vs QP, bound respecting, nullspace desat)
- Added alternating/weighted multi-vector experiments to Planner paper
- Committed all papers to repo for version control
- Research consolidated into `research/RESEARCH_MASTER.md`

**To Resume**:
1. Read `papers/mckeen_dissertation_interview_session.md` for full context
2. Check "Open Items" section for pending tasks
3. Key decisions made:
   - GENERALIZABILITY is core thesis (show diverse configs, not just BC2)
   - LP beats QP for direction preservation; "1a-Power brake only" is best constrained QP
   - BC2 parameters in `ADCS/satellite_factory/satellites/create_cubesats.py`
   - Test infrastructure: `testing/paper_todo_tests/` (90 tests, 89% coverage)

**Next Steps**:
- Expand bolt-on framework into actual paper BODY sections (currently in notes/comments)
- Run experiments for each framework stage
- Get BC2 mission data from team (pointing req, launch timeline)
- Run fresh MC experiments with diverse configs
- Validate QP constraint results with fully randomized MC

**Venues & Deadlines**:
- SmallSat Europe (May 2026): 3+1 paper, Package paper
- SmallSat USA: Planner paper, Generalized Control paper
