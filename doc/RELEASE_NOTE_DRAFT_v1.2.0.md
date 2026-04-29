# Release Note Draft: v1.2.0

This release makes the cleaned `exp5` workflow usable end to end for the main temperature downscaling path.

Highlights:

- training is now integrated into the cleaned workflow
- Grace GPU training has been validated with a documented working environment
- long-period inference, daily metrics, monthly metrics, VALUE metrics, and plot generation now work coherently together
- a masking bug in VALUE evaluation for retrained outputs has been fixed
- checkpoint bundle support has been added to improve reproducibility and portability
- public short-course material pages have been added for the EGU26 session

Practical outcome:

- the repository now supports both checkpoint reuse and retraining more honestly
- downstream validation of retrained checkpoints is scientifically coherent across train/test metrics, daily/monthly diagnostics, and VALUE-style summaries
- users are given a clearer path to understand what must remain compatible for checkpoint reuse, and when retraining is required instead
