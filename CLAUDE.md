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
**Status**: Active - ready for experiment execution

**Purpose**: Dissertation defense prep - sharpening 4 papers + dissertation through structured interview

**Key Files**:
- Interview notes: `papers/mckeen_dissertation_interview_session.md`
- Papers (Windows): `/mnt/c/Users/LV - Patrick McKeen/Writing/`
  - `3+1 Ppaer/3_MTQ___1_RW_Control_MASTER/main2.tex`
  - `Planner paper/MTQ_Planner_MASTER/main2.tex`
  - `Package paper/Generalized_ADCS_Python_MASTER/main2.tex`
  - `Generalied Control Paper/Generalized_ACS_MASTER/main2.tex`

**To Resume**:
1. Read `papers/mckeen_dissertation_interview_session.md` for full context
2. Check "Open Items" section for pending tasks
3. Key decisions made:
   - GENERALIZABILITY is core thesis (show diverse configs, not just BC2)
   - LP beats QP for direction preservation; "1a-Power brake only" is best constrained QP
   - BC2 parameters in `ADCS/satellite_factory/satellites/create_cubesats.py`
   - Test infrastructure: `testing/paper_todo_tests/` (90 tests, 89% coverage)

**Next Steps**:
- Run `pytest testing/paper_todo_tests/ -v` to validate tests
- Get BC2 mission data from team (pointing req, launch timeline)
- Decide QP constraint MC validation approach
- Run fresh MC experiments with diverse configs

**Venues & Deadlines**:
- SmallSat Europe (May 2026): 3+1 paper, Package paper
- SmallSat USA: Planner paper, Generalized Control paper
