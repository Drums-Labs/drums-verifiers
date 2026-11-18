# The task library: what makes a computer-use task good for RL

This is the rubric we hold every task family in `src/drums_tasks/families/` to, the audit of the 40 families that
existed before this change (37 written before D1 plus the three D1 templates), and the 11 families added to fill the
gaps the audit found. The D2 pool built from it is in [d2-pool.md](d2-pool.md). How to write a family is in
[CONTRIBUTING.md](../CONTRIBUTING.md); this page is about which families are worth training on.

Sources. Claims from the literature cite a page we opened for this document. Everything marked **D1** is our own
experience from the D1 run (Qwen3.5-27B, full-parameter GRPO through prime-rl, 2026-11-16): one family, 64 training
variants, the numbers below come from that run's report and run log.

## 1. The rubric

A family is good for RL when all eight hold. The first four can be checked offline (the tests do); the band (5) needs a
step-0 scan on the real model.

1. **Verifiable from the end state.** The checker reads files the task's apps write (a workbook, a document, a folder)
   after the episode and returns 1 or 0. It never reads the agent's transcript or trusts what the agent says it did.
   This is the OSWorld design: each task has "a detailed initial state setup configuration and a custom
   execution-based evaluation script" ([Xie et al. 2024, arXiv 2404.07972](https://arxiv.org/abs/2404.07972)).
   Every check kind also has a redundant twin (`check_alt`, another read path and another parser), and drumsd's
   reward audit compares the two.
2. **No shortcut satisfies the checker.** The checker pins the whole state the task is about, not just the field
   that changes: `xlsx_cells` and `xlsx_table` pin every cell of the table, `files_state` pins every file's sha256
   and the exact folder listing, `docx_paragraphs` pins every paragraph. Each variant ships **at least three named
   near misses** (scripted solutions that look right and must score 0) and a **scope probe** (the oracle plus one
   edit outside the task, which the reward audit must flag). A **no-op must score 0** (the task is not solved at the
   start; this matters most for recovery tasks). The adversarial probe list per family is in section 3. Public
   benchmarks have measured how often this goes wrong: our bench note
   [Verifiers, contamination, and why zero needs a fired positive control](https://bench.drums.run/research/verifiers-and-contamination)
   collects the reported false-positive rates (OSWorld 2.0's "scoring/environment mismatch" tags, ScaleCUA's judge
   agreement, UI-TARS-2's reward model); we cite them through that note and did not reopen the papers here.
3. **Variants are not memorisable.** Every name, number, file and phrasing is drawn from the variant's seed, and
   `params` records every drawn parameter plus a digest of the drawn content (lossless params). Between the training
   and held-out sets `python -m drums_tasks.split` checks three things and fails on any collision: the same
   instruction text, a seeded file with the same path and content, and the same params. A family whose instruction
   space is so small that wordings repeat across seeds (calc_export_csv: 24 of 32 held-out wordings also in
   training) teaches a lookup, not a skill.
4. **A stated horizon tier.** Short: 5 to 15 GUI actions (one dialog or one edit and a save). Medium: 15 to 50
   (several steps or rules stated once, which must be kept to the end). Long: 50 to 300, usually multi-app. The tier
   is the `difficulty_hint.horizon` of each new family and sets the turn cap (D1 ran at 15 turns: every Calc and
   Writer episode of the step-0 scan hit the cap).
5. **A learnable band at step 0, for the model we train.** GRPO learns only from groups with mixed outcomes: "if all
   outputs ... of a particular prompt are correct and receive the same reward, the resulting advantage for this
   group is zero. A zero advantage results in zero policy gradients" (DAPO, [Yu et al. 2025, arXiv
   2503.14476](https://arxiv.org/abs/2503.14476), section 3.2). Foster et al. find that during LLM RL "many questions
   are either solved by all attempts ... or by none" and prioritise "questions with high variance of success"
   ([arXiv 2502.12272](https://arxiv.org/abs/2502.12272)). Our target: **pooled step-0 success 0.10 to 0.80 and at
   least a quarter of variants mixed at K = 2**. Measure it with a K = 2 scan: 32 training-seed variants per
   family, two episodes each, at the exact training config (model, tools, turn cap, temperature); count pooled
   success, mixed variants (one of two), floor (zero of two) and ceiling (two of two). At group size G a variant with
   rate p gives a mixed group with probability 1 - p^G - (1 - p)^G; the scan's per-variant rates feed that estimate.
   **D1:** d1_files_collect scanned 0.629 pooled with 27 of 64 mixed, 26 ceiling and 11 floor; d1_calc_aggregate
   0.008 (1 of 64 mixed); d1_writer_style 0.000 with every episode at the 15-turn cap.
6. **Enough variants for the run.** **D1:** 64 variants of one family saturated by step 5. The share of finished
   episodes discarded because the whole group had one outcome went 59% at step 4, 93% at step 5, 98% at step 6, and
   one optimizer step cost up to 7,872 desktop episodes. A pool must hold more distinct variants than the run will
   sample groups, spread over several families so that no family's band closes at once.
7. **It exercises a named skill** beyond one menu action: spatial grounding in dense UIs (a cell, a tab, a ribbon
   control), hidden state (a mode, a hidden row or file, a dialog), self-verification (a stated count or rule the
   agent can check itself against), recovery (the desktop starts broken or half-done), constraint retention (rules
   stated once at the start, kept through many steps), and multi-app transfer of data. Families that are one menu
   action (export as PDF, export as CSV) are too narrow to train on.
8. **Reward shape: binary by default.** The checker returns 1 only when the whole end state is right. Partial credit
   is allowed only where the checker can measure the part exactly and the part cannot be had without doing the work
   (for example the fraction of correct cells in a recomputed column, never "the file exists"). OSWorld allows "a
   positive decimal under 1" for partial success (arXiv 2404.07972); none of our families uses partial
   credit yet, and D1 trained on binary rewards.

Two rules about the data:

- **Contamination.** No task from a public benchmark (OSWorld, OSWorld-Verified, WindowsAgentArena, ScaleCUA's
  evaluation sets) enters a training pool; our factory only borrows OSWorld's task format. Held-out variants come from
  a different seed and a disjoint index range for **every** family (the split's three checks must pass), and transfer
  is measured on **whole families** that are never trained on. The step-0 scan runs on training-seed variants only;
  the held-out sets are never used to choose families.
- **The process that must stay alive.** A-26: 4 of 128 D1 files episodes were invalid on the `no_app_crash` gate. A
  family states its app (`app`, `open_cmd`); a crash of that app makes the episode invalid (not scored, not trained
  on), never a 0.

### What the check kinds verify, and what they do not (A-15)

The lab verifier `drums_t6.py` is unchanged by this work (W6 is moving checkers into a sandbox and may add kinds).
Kinds and their blind spots, which every family design must work around:

| kind | verifies | does not verify |
|---|---|---|
| `files_state` | sha256 of named files, absence, directories, the exact listing of named folders | timestamps, permissions |
| `xlsx_cells`, `xlsx_table` | cell values (numbers within `tol`, strings stripped); a table trims trailing empty rows and columns | formulas versus typed values, formatting, hidden rows, other sheets |
| `xlsx_sheetnames` | the sheet names in order | sheet contents |
| `docx_paragraphs` | the non-empty paragraph texts in order | styles (a heading retyped as body text passes), empty paragraphs |
| `docx_bold` | the paragraphs and the concatenated bold text | italic, highlight, other character styles (A-15) |
| `pptx_titles` | the slide titles in order | slide bodies |
| `csv_table`, `text_file`, `zip_members`, `pdf_text` | as named | file encodings beyond UTF-8 |

Where a blind spot matters for a family, the audit names it.

## 2. The audit of the 40 existing families

Columns: horizon tier; checker kind; near misses before and after this change; whether `params` was lossy (a summary
such as `{"n": 7}` that repeats across unrelated variants); the overlap at the D1 and C1 default seeds (train 611161 indices
0..95, held-out 611162 indices 96..127) after the params fix; D1 or C1 evidence; the grade; the verdict.

Grade: **A** meets every offline criterion (1 to 4, 7); **B** one gap; **C** two or more; **P** parked (the app is not
proven on Drums: Chrome and VS Code have no checkpoint-compatibility evidence and GIMP is not in today's supported set).
The band (5) is unmeasured for every family except the D1 three.

Verdict: **keep** (in the D2 candidate set), **fix** (out of D2 until a change that alters the task bytes, not done
here), **drop** (not worth training on), **park** (stays in the code for later, out of every pool).

| family | app | tier | checker | NM | params | overlap after fix | D1 / C1 evidence | grade | verdict |
|---|---|---|---|---|---|---|---|---|---|
| files_rename | files | short | files_state | 3 | lossy, fixed | 0 | C1: not chosen, "too easy to be mixed" | B (one action) | keep |
| files_move_ext | files | short | files_state | 3 | lossy, fixed | 0 | C1 candidate | A | keep |
| files_delete_pattern | files | short | files_state | 2 -> 3 | lossy, fixed | 0 | | A | keep |
| files_mkdir_tree | files | short | files_state | 3 | lossy, fixed | 0 | | A | keep |
| files_zip | files | short | all(zip_members, files_state) | 3 | lossy, fixed | 0 | C1: one instruction collision at some seeds | A | keep |
| files_count | files | short | text_file number | 3 | lossy, fixed | 0 | C1: "hard without a shell" | B (answer is a small integer) | keep |
| files_find_copy | files | short | files_state | 3 | lossy, fixed | 0 | C1 candidate | A | keep |
| files_concat | files | short | files_state (exact bytes) | 3 | lossy, fixed | 0 | C1: "hard without a shell" | C (a GUI copy-paste that adds a trailing newline fails; one shell command) | fix: needs a newline-tolerant check kind |
| calc_aggregate | calc | short | xlsx_cells, table pinned | 3 | lossy, fixed | 1 instruction at 611162; 0 at 611169 | C1 candidate (seed 611169) | A | keep (held-out seed 611169) |
| calc_sort | calc | short | xlsx_table | 3 | lossy, fixed | 6 instruction | C1: 4 to 26 instruction collisions at every seed | C (instruction space) | fix: widen the file names |
| calc_new_column | calc | short | xlsx_table | 3 | lossy, fixed (26 params collisions before) | 0 | | A | keep |
| calc_rename_sheet | calc | short | xlsx_sheetnames | 2 -> 3 | partly lossy, fixed | 0 | C1 candidate | B (contents unchecked) | keep |
| calc_delete_rows | calc | short | xlsx_table | 3 | lossy, fixed | 0 | | A | keep |
| calc_export_csv | calc | short | csv_table | 3 | lossy, fixed | 24 instruction | | C (one menu action, tiny instruction space) | drop |
| writer_replace | writer | short | docx_paragraphs | 3 | lossy, fixed | 0 | C1 candidate | A | keep |
| writer_append | writer | short | docx_paragraphs | 3 | lossy, fixed | 0 | C1: "a whole sentence to type exactly" | B (typing, not GUI skill) | keep |
| writer_delete_para | writer | short | docx_paragraphs | 2 -> 3 | lossy, fixed | 0 | | A | keep |
| writer_bold | writer | short | docx_bold | 2 -> 3 | lossy, fixed | 0 | D1's bold template (d1_writer_style) scanned 0/64 | C (in 3 to 7 of 96 variants per seed the phrase occurs more than once, so the oracle itself scores 0; found by this audit's offline test) | fix: draw a phrase that occurs once |
| writer_export_pdf | writer | short | pdf_text | 3 | lossy, fixed | 1 instruction | C1: one instruction collision at every seed | C (one menu action) | drop |
| writer_heading | writer | short | docx_paragraphs | 2 -> 3 | lossy, fixed | 0 | C1 candidate | B (heading style unchecked) | keep |
| impress_title | impress | short | pptx_titles | 2 -> 3 | lossy, fixed | 0 | | B (bodies unchecked) | keep (Impress needs a live check first) |
| impress_delete_slide | impress | short | pptx_titles | 2 -> 3 | lossy, fixed | 0 | | B (bodies unchecked) | keep (Impress needs a live check first) |
| impress_add_slide | impress | short | pptx_titles | 2 -> 3 | lossy, fixed | 0 | | B (bodies unchecked) | keep (Impress needs a live check first) |
| multi_calc_to_text | calc + text | short | text_file | 3 | lossy, fixed | 29 instruction | | C (instruction space; one cell to read) | fix: widen the instruction |
| multi_doc_to_sheet | writer + calc | medium | xlsx_table | 3 | lossy, fixed | 12 instruction | | B (instruction space) | fix: widen the instruction |
| multi_sheet_to_doc | calc + writer | short | docx_paragraphs | 2 | lossy, fixed | 32 instruction (the wording never varies) | | C | drop (multi_csv_report replaces it) |
| multi_web_to_csv | chrome | short | csv_table | 3 | lossy | 32 instruction, 32 params | | P | park |
| multi_download_extract | chrome + files | short | files_state | 3 | lossy | 14 instruction, 26 file, 32 params | | P | park |
| chrome_open_tabs | chrome | short | chrome_tabs (no redundant twin) | 2 | lossy | 32 params | | P | park |
| chrome_download | chrome | short | files_state | 3 | lossy | 12 instruction, 32 params | | P | park |
| chrome_bookmark | chrome | short | chrome_bookmarks | 2 | lossy | 19 instruction, 32 params | | P | park |
| gimp_transform | gimp | short | image | 2 | lossy | 31 params | | P | park |
| gimp_resize_crop | gimp | short | image | 2 | lossy | 19 params | | P | park |
| gimp_export_jpeg | gimp | short | all(image, files_state) | 3 | lossy | 32 params | | P | park |
| vscode_setting | vscode | short | json_keys | 3 | lossy | 20 instruction, 5 file, 32 params | | P | park |
| vscode_fix_function | vscode | short | python_tests | 2 | lossy | 32 params | | P | park |
| vscode_rename_symbol | vscode | short | python_tests | 2 | lossy | 11 params | | P | park |
| d1_files_collect | files | short to medium (13 actions median at step 0) | files_state | 4 | lossless | 0 | D1: 0.629 pooled, 27/64 mixed; trained to 1.0 on held-out (+0.458); saturated by step 5 | A | keep (mid band; more variants than D1) |
| d1_calc_aggregate | calc | short | xlsx_cells, table pinned | 4 | lossless | 1 instruction at 611162; 0 at 611163 | D1: 0.008, 1/64 mixed (floor at 15 turns) | A | keep (hard tier; rescan at the D2 turn cap) |
| d1_writer_style | writer | short to medium | all(docx_bold, files_state) | 4 | lossless | 0 | D1: 0/64, every episode at the 15-turn cap | A | keep (hard tier; rescan at the D2 turn cap) |

**Headline: 21 keep, 5 fix, 3 drop, 11 park.** All 37 pre-D1 families had lossy params; the 26 on apps Drums runs
today now record every drawn parameter (the 11 parked ones were left as they are), and 8 families got a third near
miss (7 kept ones and writer_bold). With lossless params the params collisions at the default seeds went from up to
32 per family (of 32 held-out variants) to 0 for all 26; what remains are instruction collisions in the families
marked fix or drop. The tasks themselves did not change by one
byte (`tests/test_library.py` pins the sha256 of the first eight tasks of each family at seed 20261108 against 0.2.2),
so the C1 pool and every validated set stay valid; `GENERATOR_VERSION` is `t6-factory-3` because `solutions.jsonl`
changed.

What the audit found missing: every pre-D1 family is short (most are one edit and a save); nothing tests hidden
state, recovery or a rule kept over many steps; the multi-app families either use Chrome (parked) or have an
instruction space too small to hold out. Section 3 fills those gaps.

## 3. New families (11), with their adversarial probes

All are in the factory with an oracle, at least three near misses, lossless params, 96 variants at scale 1, and offline
tests (`tests/test_library.py`: oracle 1.0, every near miss 0.0 and a no-op 0.0 under both checkers; zero overlap at
611161/611162, and at the D2 seeds 611171/611172 by the check in d2-pool.md; the office oracles also through a real `soffice` when installed). They use existing check kinds only.
None has run on a live desktop yet: `drums_tasks.validate` must accept every variant before a freeze (d2-pool.md).

| family | tier | skills | the task | near misses (each scores 0) | other probes to run live |
|---|---|---|---|---|---|
| medium_files_rules | medium (9 to 14 files) | constraint retention, rule precedence | sort an inbox into three new folders by four rules stated once, one of which overrides the others | draft rule ignored; .log files kept; one file left behind; copied not moved | create the folders and move everything into one; rename files to fit the rules |
| medium_calc_pipeline | medium | constraint retention, dense UI | delete rows by a threshold, add a computed column, sort by it, add a total row after a blank row, rename the sheet | total computed before the delete; boundary rows deleted; sorted the other way; sheet not renamed; no rows deleted | values typed as text; the table rebuilt in a new sheet; formulas that read deleted cells (#REF!) |
| medium_writer_restructure | medium | ordering, constraint retention, dense UI | delete a section, rename a heading, then order the sections A to Z by their new headings, text moving with its heading | ordered by the old headings; headings sorted without their text; section not deleted; heading deleted but its text kept; not renamed | heading styles lost (passes: docx_paragraphs is style-blind, see section 1) |
| multi_csv_report | medium | multi-app, aggregation | CSV into a workbook, then a Writer report: per-category totals largest first and a grand total | alphabetical order; report only, no workbook; grand total missing; last row dropped from both | the CSV edited; totals typed from a guess; the workbook with the header as data |
| multi_letter_fill | medium | multi-app, lookup, Save As | fill a letter template's four placeholders from one customer's row and save it under a new name | values from the neighbouring row; one placeholder left; saved over the template; saved under both names; balance reformatted (1,234.00) | the template's other text edited |
| hidden_files_dotfiles | short | hidden state, self-verification | move every file of one type into a new folder; some are dotfiles the file manager hides; the instruction states the count | visible files only; leading dot dropped; copied not moved; a hidden file of another type moved too | moving the whole folder |
| hidden_calc_rows | short | hidden state, dense UI | an aggregate over every data row of a sheet in which 2 to 5 rows are hidden (the extreme row is hidden for maximum and minimum) | visible rows only; hidden rows deleted; wrong cell; wrong aggregate | SUBTOTAL(109, ...), which skips hidden rows (must score 0, and does) |
| hidden_writer_readonly | short | hidden state, dialog | a Find and Replace in a document that opens in read-only viewer mode; the result must be saved in place | saved as a copy; first occurrence only; another word changed | live check: confirm `--view` shows the read-only bar on the Drums image |
| recover_files_interrupted | medium | recovery, self-verification | finish an interrupted move: some files moved, some not, one copy in the target truncated, a .part and a "(copy)" leftover | the truncated copy kept; .part left; duplicate left; copied not moved; other types moved too | deleting the source folder |
| recover_calc_column | medium | recovery, self-verification | a computed column with wrong numbers, blanks and text placeholders in 3 to 5 rows; fix only the bad cells | blanks and text fixed, numbers not; numbers fixed, blanks not; bad rows deleted; an input column changed instead | the whole column retyped (passes when right: allowed) |
| recover_writer_steps | short to medium (6 to 12 steps) | recovery, ordering | numbered steps shuffled by 2 or 3 swaps with one step pasted twice; restore Step 1..N once each | duplicate left; order not fixed; one swap left; both copies deleted; sorted as text (Step 10 before Step 2) | retyping a step with a small wording change |

Families still missing, for a later round: long-horizon (50 to 300 actions) multi-app tasks (a month-end close that
touches five files), Impress-based multi-app tasks, a text-editor family (none exists yet), and partial-credit variants of the recovery families once a check kind can count correct cells.

## 4. Future work: a CAD family set (FreeCAD)

Not built; what it would need for a possible later CAD model.

- **The app on Drums.** FreeCAD (Qt plus Coin3D OpenGL, software rendering on the desktop) must pass the
  checkpoint-compatibility matrix first: a GL context under CRIU restore is unproven, like Chrome.
- **Checkers through FreeCAD's Python API**, run headless (`freecadcmd`) inside the sandboxed checker runtime W6 is
  building (the checker venv cannot import FreeCAD today, so this needs a new check kind and its redundant twin):
  open the `.FCStd`, recompute, and compare geometry within tolerances (volume, surface area, bounding box, centre of
  mass, face and edge counts, `isValid()` of the solid), plus the feature tree (feature names and types, sketch
  constraint counts, a Pad length or Pocket depth). The redundant twin parses the `.FCStd` zip's `Document.xml`
  directly and checks the same parameters, or reads an exported STEP through a second kernel.
- **Anti-hack.** The Python console and macros inside FreeCAD can write any model; checkers must compare geometry, not
  only parameters, and near misses include right parameters on the wrong face, mirrored features, a non-solid shell
  with the right bounding box, and a model rebuilt from primitives without the required features.
- **Recovery via injected faults** made by the setup through the API: an over- or under-constrained sketch, a
  feature that references a deleted face (the topological naming problem), a wrong value in a spreadsheet that drives
  the model, a broken Boolean, a body hidden in the tree.
- **Hidden state.** An open sketch edit mode, the wrong workbench, units set to inches, a hidden body.
- **Partial credit** is natural here (for example the volume of the symmetric difference over the target volume), but
  it invites shapes that fill the target's envelope; keep it binary until a near-miss suite shows the shaped reward
  cannot be gamed.
