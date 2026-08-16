# Dissertation Draft

A working draft, split by chapter, built from the actual implementation
and experiments in this repository, and structured to match the
University of Surrey MSc Data Science (COMM002) format: two real
COMM002 submissions were reviewed to derive this structure, notably the
8-chapter body (Introduction through Conclusion, with Results and its
critical review as two separate chapters, and a dedicated chapter
reviewing the project against its own stated objectives) rather than a
more generic thesis layout.

Two different levels of finish, deliberately:

- **Fully drafted** (03, 04, 05): Methodology, Implementation, and
  Presentation of Results. These describe work that was actually done,
  so they're written as complete prose you can edit for voice and
  emphasis, not as a skeleton.
- **Scaffolded, not written** (01, 02, 06, 07, 08): Introduction,
  Literature Review, Critical Review of the Results, Critical Review of
  the Project Objectives, and Conclusion. These are outlines with the
  key points, facts, and structure you need, deliberately left for you
  to write as prose. A dissertation is assessed on this part being your
  own scholarly argument and your own reflection on your own work, not a
  description of what a pipeline did, so filling these in yourself is
  the actual work, not busywork to route around. Chapter 7 especially:
  it's a personal assessment against your own objectives, not a
  technical summary, and can't honestly be written by anyone but you.

## Files

| File | Section | Status |
|---|---|---|
| `00a_title_page.md` | Title page | Placeholders for name/supervisor |
| `00b_declaration_of_originality.md` | Declaration of Originality | Standard-form draft, check against current official template |
| `00c_abstract.md` | Abstract | Full draft |
| `00d_acknowledgements.md` | Acknowledgement(s) | Yours to write |
| `00e_table_of_contents.md` | Table of Contents | Skeleton, fill in page numbers last |
| `00f_list_of_figures.md` | List of Figures | Skeleton, candidate figures listed |
| `01_introduction.md` | 1. Introduction | Scaffold (Aims and Objectives fully drafted, rest is scaffold) |
| `02_literature_review.md` | 2. Literature Review | Scaffold |
| `03_methodology.md` | 3. Methodology | Full draft |
| `04_implementation.md` | 4. Implementation | Full draft |
| `05_presentation_of_results.md` | 5. Presentation of Results | Full draft |
| `06_critical_review_of_results.md` | 6. Critical Review of the Results | Scaffold |
| `07_critical_review_of_project_objectives.md` | 7. Critical Review of the Project Objectives | Scaffold |
| `08_conclusion.md` | 8. Conclusion | Scaffold |
| `09_references.md` | References | Full draft for citations `[1]`-`[5]`, extend if the lit review gap is filled |

## How to use this

1. Read Chapters 3-5 first. They're the evidentiary backbone everything
   else refers back to, and they're the closest to submission-ready.
   Every number in them traces back to a file in `results/` or
   `experiments/*/results/` (referenced inline), and
   `notebooks/results_summary.ipynb` reproduces most of the tables in one
   place if you want to regenerate a figure or check a number.
2. Read Chapter 6 next: it's a scaffold, but a dense one, it already
   contains the specific analytical points (which results qualify which
   claims, where the evidence is weak vs. strong) that Chapters 3-5 set
   up. Turning it into prose is mostly a writing task at that point, not
   a re-analysis task.
3. Chapter 1's Aims and Objectives (§ 1.3) are drafted in full, since
   Chapter 7 is assessed directly against them. Read and edit that list
   *before* writing Chapter 7, if you change an objective, Chapter 7's
   assessment of it needs to change too.
4. Write Chapters 1, 2, 7, 8 last, roughly in that order. Introduction
   and Conclusion will naturally echo each other once both exist; it's
   normal to draft Conclusion, revise Introduction, and repeat.
5. `docs/methodology.md` and `docs/literature_comparison.md` (one level
   up) are the working notes Chapters 3 and 2/6 were built from; the
   literature comparison doc has more detail on the Hoang et al.
   replication than made it into Chapter 5/6, useful if you want to go
   deeper in the critical review.
6. Front matter (`00a`-`00f`): fill in your name, supervisor, and
   acknowledgements; check the Declaration of Originality against your
   department's current official wording before submission, it's a
   standard-form draft, not verified against this year's template.

## Rough page budget

Matches the ~60-page target discussed earlier in the project, adjusted
for the 8-chapter structure.

| Chapter | Pages |
|---|---|
| 1. Introduction | 4-6 |
| 2. Literature Review | 10-14 |
| 3. Methodology | 8-10 |
| 4. Implementation | 6-8 |
| 5. Presentation of Results | 12-16 |
| 6. Critical Review of the Results | 6-8 |
| 7. Critical Review of the Project Objectives | 3-4 |
| 8. Conclusion | 3-4 |
