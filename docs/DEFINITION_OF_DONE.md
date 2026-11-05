# Definition of Done — Measure It

**Assessment:** xis.ai AI / Computer Vision Technical Assessment, Version 2.0

This checklist is not complete until every checked item has a number, file, command output, commit hash or link as evidence. Unfinished requirements must remain unchecked and must be listed honestly.

## A. Planning and Git

- [ ] Public GitHub repository created — evidence: ____________________
- [ ] Branch `dev-test01` exists — evidence: ____________________
- [ ] PLAN.md committed before implementation code — commit: ____________________
- [ ] Fixed test split manifest committed before implementation code — commit: ____________________
- [ ] Meaningful progressive commits exist for each major stage — evidence: ____________________
- [ ] No single final commit containing the entire project — evidence: ____________________

## B. Camera calibration

- [ ] At least 20 valid checkerboard/ChArUco photographs — count: ______
- [ ] Actual checkerboard square size measured and recorded — evidence: ____________________
- [ ] OpenCV calibration executes successfully — command/evidence: ____________________
- [ ] Camera matrix saved — file: ____________________
- [ ] Distortion coefficients saved — file: ____________________
- [ ] Reprojection error reported — value: ____________________
- [ ] An original image and its undistorted result are saved — evidence: ____________________

## C. Dataset and labels

- [ ] At least 70 self-captured target-object images approved — count: ______
- [ ] Approved images are undistorted — evidence: ____________________
- [ ] Every approved image has a valid segmentation mask — count: ______
- [ ] Train/validation/test manifest is committed — commit: ____________________
- [ ] Split counts are reported — train: ______ val: ______ test: ______
- [ ] Test images were not used for training, tuning or checkpoint selection — evidence: ____________________

## D. Segmentation model

- [ ] Allowed segmentation architecture selected and justified — evidence: ____________________
- [ ] Model trained or fine-tuned on the captured dataset — training run: ____________________
- [ ] Random seed and training configuration saved — file: ____________________
- [ ] Test mask IoU reported — value: ____________________
- [ ] mAP@0.5 reported — value: ____________________
- [ ] mAP@0.5:0.95 reported — value: ____________________
- [ ] Precision and recall reported — values: ____________________
- [ ] Model weights saved or linked — location: ____________________
- [ ] SHA-256 of model weights recorded — hash: ____________________

## E. Image-to-millimetres inference

- [ ] One command accepts a new image — command: ____________________
- [ ] The image is undistorted before inference — code location: ____________________
- [ ] Target object is segmented — evidence: ____________________
- [ ] ArUco reference is automatically detected — evidence: ____________________
- [ ] Physical reference dimensions are recorded — dimensions: ____________________
- [ ] Coplanar reference geometry is used for metric conversion — code location: ____________________
- [ ] Width in millimetres is reported — output: ____________________
- [ ] Height in millimetres is reported — output: ____________________
- [ ] Confidence is reported — output: ____________________
- [ ] Mask overlay is saved — file: ____________________
- [ ] Object-not-found case is handled — evidence: ____________________
- [ ] Reference-not-found case is handled — evidence: ____________________

## F. Physical accuracy

- [ ] At least 10 physical instances are measured manually — count: ______
- [ ] Measuring instrument and resolution are recorded — evidence: ____________________
- [ ] System results are saved for the same instances — evidence: ____________________
- [ ] Per-instance width and height table is complete — file: ____________________
- [ ] MAE reported — value: ____________________
- [ ] Median error reported — value: ____________________
- [ ] Worst-case error reported — value: ____________________
- [ ] Mean percentage error reported — value: ____________________

## G. Required experiments and refusal behavior

- [ ] Same instances evaluated using raw and undistorted images — evidence: ____________________
- [ ] Raw and undistorted errors reported side by side — evidence: ____________________
- [ ] Difference explained using measured results — evidence: ____________________
- [ ] At least one speed or accuracy objective measured — evidence: ____________________
- [ ] One selected failure condition tested at increasing severity — evidence: ____________________
- [ ] Error change under that condition reported — evidence: ____________________
- [ ] Refusal threshold defined — value: ____________________
- [ ] Low-confidence/unreliable measurement refusal demonstrated — evidence: ____________________
- [ ] Number of refused images reported — count: ______
- [ ] Engineering decision record completed — location: ____________________

## H. Reproducibility and documentation

- [ ] PLAN.md complete and accurate
- [ ] OBJECTIVES.md contains targets and actual results
- [ ] DEFINITION_OF_DONE.md contains evidence for completed items
- [ ] README has installation and run instructions
- [ ] Python and library versions pinned
- [ ] CPU, GPU, RAM and OS recorded
- [ ] Camera model and actual image resolution recorded
- [ ] Training random seed and configuration saved
- [ ] No dead code, unexplained files or commented-out blocks
- [ ] Clean-environment reproduction attempted — evidence: ____________________

## I. Recording and submission

- [ ] Loom recording is 5 minutes or less
- [ ] A new photograph is processed on camera
- [ ] Physical ruler/calliper comparison is shown
- [ ] Code is followed from input to millimetre output
- [ ] K1 answered correctly
- [ ] K2 answered correctly using actual results
- [ ] K3 answered honestly
- [ ] K4 answered honestly
- [ ] Public repository is complete
- [ ] Pull request from `dev-test01` into `main` created — URL: ____________________
- [ ] Loom URL available — URL: ____________________
- [ ] Both links submitted by the deadline — evidence: ____________________

## Final status

**Completed requirements:** ______
**Incomplete requirements:** ______
**Known limitations:** ____________________________________________
**Reason for stopping:** __________________________________________
**Final repository commit:** _______________________________________
