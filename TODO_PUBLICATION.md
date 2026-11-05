# Publication TODO

Tracked in-repo. One item checked at a time.

## Phase 0 — Setup
- [x] 0.1 TODO_PUBLICATION.md
- [x] 0.2 paper/ directory
- [x] 0.3 paper/analysis/ + paper/results/

## Phase 1 — Statistical rigor
- [x] 1.1 Mann-Whitney U + KS + effect size (GITI vs MRC PET)
- [x] 1.2 Bootstrap CIs (10k resamples)
- [x] 1.3 Severity proportion comparison
- [x] 1.4 Conflict-type distribution comparison
- [x] 1.5 Power analysis
- [x] 1.6 paper/analysis/statistical_tests.py + paper/results/stat_tests.json
- [x] 1.7 Draft Results 3.1 + 3.2

## Phase 2 — Detection validation
- [ ] 2.1 Sample 100 stratified frames
- [ ] 2.2 Hand-annotate
- [ ] 2.3 Per-class P/R/F1
- [ ] 2.4 RT-DETR + YOLOv11n baseline
- [ ] 2.5 detection_metrics.py + detection_metrics.json
- [ ] 2.6 Methods 2.2 + Results 3.3

## Phase 3 — Ground-truth PET validation (critical path)
- [ ] 3.1 Annotation protocol
- [ ] 3.2 Two annotators, 40 sampled events
- [ ] 3.3 40 sampled non-events (false negatives)
- [ ] 3.4 Cohen kappa
- [ ] 3.5 Precision/recall/F1 vs consensus
- [ ] 3.6 Per-conflict-type agreement
- [ ] 3.7 gt_validation.py + gt_validation.json
- [ ] 3.8 Results 3.4 + Limitations

## Phase 4 — Baseline comparison
- [ ] 4.1 Tracker-only (yolo-cpu)
- [ ] 4.2 RT-DETR-only
- [ ] 4.3 Compare counts/distributions/agreement
- [ ] 4.4 baselines.py + baselines.json
- [ ] 4.5 Results 3.5

## Phase 5 — Manuscript
- [ ] 5.1 Introduction
- [ ] 5.2 Related Work
- [ ] 5.3 Methods 2.1-2.5
- [ ] 5.4 Results 3.1-3.5
- [ ] 5.5 Discussion
- [ ] 5.6 Conclusion
- [ ] 5.7 Reproducibility appendix
- [ ] 5.8 Final figures
- [ ] 5.9 Document 0.857 retraction

## Phase 6 — JOSS
- [ ] 6.1 Scope fit
- [ ] 6.2 joss_paper.md
- [ ] 6.3 paper.bib
- [ ] 6.4 Submission checklist
- [ ] 6.5 Submit

## Phase 7 — Full paper
- [ ] 7.1 Target venue
- [ ] 7.2 Format
- [ ] 7.3 Internal review
- [ ] 7.4 Cover letter
- [ ] 7.5 Submit
