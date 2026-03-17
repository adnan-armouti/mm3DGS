# Tips for Working Effectively with Claude Code

Based on our collaboration on the mmIR/mm3DGS project, here are concrete ways to get the most out of Claude Code.

---

## 1. Use CLAUDE.md Files (You're Doing This Now)

The `.claude/CLAUDE.md` file is loaded into every conversation automatically. Put your project's critical invariants here:
- Environment setup (conda env, Python path)
- Architectural decisions that are easy to forget
- Known footguns (like the DrJit double-set_variant segfault)
- File layout conventions

**Tip**: Keep it under 200 lines. If it's too long, Claude skims it. Be specific and actionable.

---

## 2. Start with a Plan Before Coding

For any non-trivial task, ask Claude to write a plan as a `.md` file first (like we did with `PLAN_sc_alignment_via_cascade_trajectory_transfer.md`). This:
- Forces alignment before code is written
- Lets you catch wrong assumptions early (much cheaper than debugging)
- Creates documentation as a side effect

**Pattern**: "Can you please provide a plan to implement this as a .md file saved to md/? I will review it before approving for execution."

---

## 3. Parallelize with Both GPUs

When you have independent work across scenes/configs, tell Claude you have multiple GPUs:
- "I have access to two 4090 GPUs — please parallelize"
- Claude will split work across `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`
- This cut our alignment time from ~2.5 hours to ~1.25 hours

---

## 4. Ask for Visualizations Early

Don't wait until the end to check if results make sense. Ask for:
- RA image comparisons (rendered vs GT) after alignment
- Side-by-side before/after plots
- Per-scene metric tables

"Can you save visualizations so I can inspect the rendered RA images against GT?" catches bugs that aggregate numbers hide.

---

## 5. Be Explicit About What NOT to Overwrite

Claude is careful by default, but when copying/moving data, say it explicitly:
- "Please make sure not to overwrite any existing files or output directories"
- "Do not modify the original files"

---

## 6. Use the /commit Skill for Git

Instead of manually staging and committing, use `/commit` — Claude will:
- Check `git status` and `git diff`
- Draft an appropriate commit message
- Stage only relevant files (avoiding secrets, large binaries)

---

## 7. Chain Experiments with Clear Before/After

When testing a hypothesis (like "do trained materials improve alignment?"), structure it as:
1. State the hypothesis
2. Ask Claude to run the experiment
3. Ask for a comparison table

This is what we did when comparing default-material vs trained-material refinement — the table format made the answer immediately clear.

---

## 8. Use Memory for Cross-Session Context

Claude Code has persistent memory (`.claude/projects/.../memory/`). Important facts that should survive across sessions:
- "The 2n+1 formula maps cascade frame N to SC frame 2N+1" (later proved wrong, but good to record)
- "Trained-material refinement gives 0.464 mean CC"
- "Default-material refinement is better (0.589) for alignment"

**Pattern**: "Please remember that..." or Claude will auto-save things that seem important.

---

## 9. Ask Claude to Audit Before Shipping

Before finalizing a standalone submission:
- "Can you make sure submission_v2 has no references to paths outside of itself?"
- "What other input data files do we need?"

This caught 4 hardcoded dataset paths and 3 stale antenna pattern paths that would have broken the submission.

---

## 10. Use Subagents for Deep Exploration

For broad codebase searches ("find all files that import X", "diff these two scripts"), Claude can launch specialized subagents that work in parallel. You don't need to ask for this — Claude does it automatically for complex searches. But you can encourage it:
- "Please analyze both the old and new versions in parallel"

---

## Common Mistakes to Avoid

1. **Don't re-explain context Claude already has** — if it's in CLAUDE.md or memory, Claude knows it
2. **Don't ask Claude to "be careful"** — instead, state the specific constraint ("don't delete X")
3. **Don't run the same failing command repeatedly** — ask Claude to diagnose root cause instead
4. **Don't manually copy-paste code between files** — ask Claude to do it (it tracks all the import changes)
5. **Don't forget to invalidate cached renders** — when alignment configs change, old `adc_rendered_single_chip.npy` files are stale
