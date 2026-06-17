# EGU26 Short Course Session Summary

> Release compatibility: this EGU26 short-course material is maintained against `idownscale` release `v1.4.0`. If you are using another release, check that workflow runner names, paths, and expected outputs still match that version.

This short course introduces robust machine-learning methods for statistical downscaling of coarse-resolution climate model scenarios.

Using daily near-surface temperature as a case study, the session follows the full workflow from pre-Phase-1 France target preparation and data cropping through bias correction, training, inference, and evaluation. The emphasis is practical and climate-oriented: how to prepare consistent predictor and target datasets, when retraining is required, how bias correction interacts with machine learning, and how to evaluate downscaling results beyond a single scalar score.

The course is designed for climate scientists, impact researchers, and technical users who want a clear and reproducible introduction to machine-learning-based downscaling. Prior expertise in machine learning is not required, although familiarity with climate data and downscaling concepts is helpful.

The practical material is organized so that students can return to it after the
session and recover the full workflow, data paths, and validation logic from the
GitHub pages and notebook alone. The notebook is the main hands-on experience,
while the companion Markdown pages provide more detailed reference material.

For reproducibility, students should use the repository at tag `v1.4.0` when
following these notes. The short course is maintained against that release even
though the repository may continue to evolve afterward.

## Start here

Students should begin from the GitHub version of this folder and use:

- [Session materials](./SESSION_MATERIALS.md)
- [Data setup quickstart](./DATA_SETUP_QUICKSTART.md)
- [Notebook](./egu26_short_course_notebook.ipynb)
- [Environment setup](./ENVIRONMENT_SETUP.md)
- [Dataset files to provide](./DATASETS_TO_PROVIDE.md)
- [Presentation PDF](./EGU26_CPAGE_ILAZIC_MTOSIC.pdf)

## Mercure publication

The published course package is hosted on Mercure at:

- `https://mercure.cerfacs.fr/egu26scml/`

The current release is organized as:

- `required/`
- `nice_to_have/`
- `raw_data/`
- `phase_outputs/`
- `egu26_sc_required.tar.gz`
- `egu26_sc_nice_to_have.tar.gz`

The `idownscale` repository remains the reference for code, workflow scripts, and editable documentation. Mercure is the canonical download location for attendee-facing files.

## What the session provides

Attendees should be able to retrieve, from Mercure alone:

- the notebook in `required/notebook/egu26_short_course_notebook.ipynb`
- one portable checkpoint bundle in `required/checkpoint_bundles/exp5_unet_all_bundle/`
- one historical prediction example in `required/predictions/`
- summary metrics in `required/metrics/`
- selected validation figures in `required/plots/`
- the climate inputs and derivatives needed for the workflow in `raw_data/`, `nice_to_have/`, and `phase_outputs/`
- the packaged downloads `egu26_sc_required.tar.gz` and `egu26_sc_nice_to_have.tar.gz` for easier bulk retrieval

The notebook is intended both to be read and to be re-executed. Attendees should be able to follow the full sequence from directory preparation and pre-Phase-1 France cropping through the later workflow phases, running each phase independently when needed. Some phases can be long, on the order of several hours, so the notebook should support both careful reading and practical reruns phase by phase. The same workflow should remain usable on a laptop, workstation, or supercomputer, with the practical difference being how much data is processed and how long the long-running phases take. Users can reuse the published checkpoint when their setup remains compatible, or optionally retrain when they change the configuration, domain, predictors, preprocessing, normalization, or other key parts of the workflow setup.

This split keeps the course material reproducible without requiring attendees to mirror an internal HPC filesystem layout.

## Learning goals

By the end of the session, students should be able to:

- explain why the downscaling problem is not solved by coarse climate data alone
- describe the logic of the France `exp5` workflow from preparation to evaluation
- identify which phases can be run independently and how to verify them
- understand when published outputs are sufficient for demonstration and when a live rerun is useful
- understand when checkpoint reuse is acceptable and when retraining becomes necessary
- navigate from the landing-page material to the runnable notebook and phase checks without relying on private context

## Follow-up contact

For follow-up questions after the short course:

- LinkedIn: `https://www.linkedin.com/in/pagechristian`
