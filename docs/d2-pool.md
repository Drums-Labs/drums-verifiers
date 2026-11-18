# The D2 task pool: proposal

D2 reruns RL on the same Qwen3.5-27B as D1 with about 500 USD of Modal GPU time and is compared against D1. D1 trained
on 64 variants of one family and saturated by step 5 (see [task-library.md](task-library.md), rubric item 6). This
proposal replaces that with several hundred variants over many families, chosen on the real model by a step-0 scan
before the freeze. Nothing here is built yet; the artifacts that need a host (live validation, the scan) are listed as
steps, not results.

## 1. Candidates and difficulty tiers

The scan's candidate set is the 21 families the audit keeps plus the 11 new ones: 32 families.

| horizon tier | candidates |
|---|---|
| short (5 to 15 actions) | files_rename, files_move_ext, files_delete_pattern, files_mkdir_tree, files_zip, files_count, files_find_copy, calc_aggregate, calc_new_column, calc_rename_sheet, calc_delete_rows, writer_replace, writer_append, writer_delete_para, writer_heading, impress_title, impress_delete_slide, impress_add_slide, d1_calc_aggregate, hidden_files_dotfiles, hidden_calc_rows, hidden_writer_readonly |
| medium (15 to 50) | d1_files_collect, d1_writer_style, medium_files_rules, medium_calc_pipeline, medium_writer_restructure, multi_csv_report, multi_letter_fill, recover_files_interrupted, recover_calc_column, recover_writer_steps |
| long (50 to 300) | none yet (task-library.md, end of section 3) |

The second tier axis comes from the scan, per family, on the D2 model and config: **easy** (pooled step-0 success
above 0.80), **learnable** (0.10 to 0.80 with at least a quarter of variants mixed), **hard** (below 0.10). Only
learnable families train. Hard ones are the next run's material (a longer turn cap, a larger model, or an easier
sibling first); easy ones are kept as a small regression check in evaluation, not trained on.

## 2. The pool

Numbers assume about 14 families pass the scan; the rules below decide if more or fewer do.

| set | families | variants per family | seed | indices | total |
|---|---|---|---|---|---|
| training | the learnable families, at most 3 per app and at least 4 medium-tier, at least 2 hidden-state and 2 recovery families if they pass | 48 | 611171 | 200..247 | about 672 |
| held-out by seed | every training family | 16 | 611172 | 248..263 | about 224 |
| held-out by family (transfer) | 2 whole families, never trained | 32 | 611173 | 300..331 | 64 |
| D1 anchor | d1_files_collect, D1's own held-out set | 32 | 611162 | 64..95 | 32 |

- **Why 48 per family.** D1's run admitted about 96 groups (6 steps of 16 groups of 8) and discarded most of the rest
  as zero-advantage. With about 672 training variants and a similar or somewhat larger number of steps, each variant
  is sampled about once: a variant cannot be memorised, and saturation, if it comes, shows up per family (watch the
  per-family batch reward each step) rather than across the whole pool at once.
- **Index ranges start at 200** so no D2 task id repeats a D1 or C1 id (the decoy file carries the id, and the split's
  file check would flag it). `python -m drums_tasks.split --start 200` does this (new in this change).
- **Transfer families.** Proposed: **recover_writer_steps** and **multi_letter_fill**. Their skills (recovery,
  multi-app lookup) and apps (Writer, Calc) are trained through other families (recover_files_interrupted,
  recover_calc_column, multi_csv_report), so a gain on them measures transfer to a new task, not to a new app. Both
  must be learnable at step 0 to measure a gain; if one is not, take the next in the same role
  (recover_calc_column, then hidden_writer_readonly).
- **D1 anchor.** D1's 32 held-out variants of d1_files_collect, evaluated at K = 6 as in D1, so D2's policy is compared
  with D1's on exactly the set D1 was judged on. If d1_files_collect trains in D2, its D2 training variants must not
  overlap that set (checked below).
- **Seeds checked offline.** For every one of the 32 candidates (and writer_bold), 48 training variants at 611171 (indices 200..247)
  against 16 held-out at 611172 (248..263): zero collisions on instruction, file and params, all of them (checked
  2026-11-17 with `drums_tasks.split.overlaps` at generator `t6-factory-3`). d1_files_collect's D2 training variants
  against D1's held-out set, and D1's training variants against D2's held-out set: zero collisions.
- **Sampling.** Uniform over the training variants (prime-rl's default), as in D1. The pool composition is the lever
  against saturation; an online curriculum stays off unless it is tested before the freeze (D1's 11:32Z decision).

## 3. The step-0 scan that calibrates it

Run on the real model before the freeze, on training-seed variants only (indices 200 onward), never on held-out sets.

1. **Validate first.** `drums_tasks.validate` on a Drums host for every variant of every candidate: oracle 1.0 trusted,
   every near miss 0.0, scope probe flagged. On the same pass confirm what only a live desktop shows: LibreOffice
   Impress is on the image; `libreoffice --writer --view` opens read-only with the Edit Document bar
   (hidden_writer_readonly); the file manager hides dotfiles by default (hidden_files_dotfiles); hidden rows render
   hidden (hidden_calc_rows); a CSV opened in Calc shows the Text Import dialog (multi_csv_report). A family that
   fails any of these is out.
2. **Config.** Exactly the D2 training config minus the trainer: Qwen3.5-27B, the same tools, temperature 0.8,
   thinking off, the same history settings (W5), and the D2 turn cap. The cap must fit the medium tier: **40 turns**
   is proposed (D1's 15 put every Calc and Writer episode at the cap). Short tasks end early, so the cap costs only
   on the episodes that fail.
3. **Stage 1.** 16 variants x K = 2 per family, 32 families: 1,024 episodes. D1's scan ran 384 episodes in 34 minutes
   at 48 desktops in flight for about 5.5 USD of GPU (RUNLOG 11:27Z); at a 40-turn cap expect roughly 2 to 3 times
   that per episode, about 30 to 45 USD for stage 1 (an estimate; the Modal invoice is the record).
4. **Stage 2.** 16 more variants x K = 2 for the families within 0.1 of a band edge or with 3 to 5 mixed of 16, to
   firm up the decision: at most about 400 episodes.
5. **Record per family:** pooled success with a 95% Wilson interval, mixed / floor / ceiling counts, invalid rate
   (`no_app_crash` and the rest), median turns and wall time per episode, and from the action transcripts (A-42, W5's
   DRUMS_RECORD change) three successful episodes read by hand for a shortcut the checker missed.
6. **Admission rule** (fixed before the scan runs): pooled success 0.10 to 0.80, at least 25% of scanned variants
   mixed, invalid rate at most 5%, no shortcut found. Then, over the chosen pool, the expected share of zero-advantage
   groups at G = 8 (from per-variant rates shrunk to the family mean, sum of p^8 + (1 - p)^8) should be at most 50%;
   if it is higher, drop the family nearest the ceiling and recompute.
7. **Freeze.** Build each training family with `drums_tasks.split --train-seed 611171 --heldout-seed 611172 --train 48
   --heldout 16 --start 200`, and each transfer family with `drums_tasks.split --train-seed 611174 --heldout-seed
   611173 --train 32 --heldout 32 --start 268`, keeping only its held-out half (indices 300..331; the training half is
   discarded unseen). Record every `pool.json` sha256 in the pre-registration, then validate the built sets once more.

## 4. What D2 measures

- Primary, as in D1: held-out by seed, paired over variants, step 0 against the last step, K = 6.
- Transfer: the two held-out families, same statistic.
- The D1 anchor: D2's final policy against D1's final policy on D1's held-out set.
- Descriptive: the zero-advantage discard share per step (D1: 59%, 93%, 98% at steps 4 to 6), and per-family reward.

## 5. Open items before the freeze

- calc_aggregate and d1_calc_aggregate need other held-out seeds at the older default 611162 (one instruction
  collision each); at 611171/611172 they have none. If the pool's counts change (more than 48 + 16 variants), rerun
  the check; a family with a collision gets the first collision-free held-out seed from a sweep, recorded in its
  `pool.json`.
- The 11 new families have never run on a desktop; step 3.1 is their first live test.
- Impress, `--view` and the Text Import dialog are unverified on the Drums image (step 3.1).
- The turn cap of 40 and the scan budget are proposals for the owner.
