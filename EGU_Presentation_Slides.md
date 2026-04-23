# EGU26 Presentation: Prototyping a Robust DL Pipeline for Climate Downscaling

**Duration:** 60 Minutes (~30 Slides)  
**Session:** Hybrid Short Course  
**Conveners:** Christian Pagé, Irida Lazić, Milica Tošić  

---

### Slide 1: Title Slide
*   **Title:** Using Machine Learning to downscale climate scenarios
*   **Subtitle:** A Robust Statistical Downscaling Framework for Regional Adaptation
*   **Presenter:** **Christian Pagé**, Research Engineer at Cerfacs, Toulouse, France
*   **Co-Authors:**  
    *   **Irida Lazić**, PhD, University of Belgrade, Serbia  
    *   **Milica Tošić**, University of Belgrade, Serbia
*   **Credits:** Main codebase and methodology developed by **Zoé Garcia**
*   **Event:** EGU26 General Assembly, Vienna, Austria | May 2026

**Speaker Notes:**  
Welcome everyone. Today we are bridging the gap between Global Climate Models and local adaptation needs using Deep Learning. This workshop is designed for both researchers and practitioners.

---

### Slide 2: The Resolution Gap (2 mins)
*   **The Problem:** GCMs operate at ~100-150 km. 
*   **The Need:** Local adaptation requires <10 km.
*   **Challenges:**
    *   Topography-induced temperature gradients.
    *   Coastline effects.
    *   Urban heat islands.
*   **Goal:** Reconstruct spatial variability hidden in coarse pixels.

---

### Slide 3: The "Perfect Model" Strategy (3 mins)
*   **The Concept:** Derived from Baño-Medina et al. (2020).
*   **Key Idea:** We cannot train on GCM-Observation pairs because their weather chronologies (dates) don't match.
*   **Solution:** Use **ERA5 Reanalysis** as a laboratory.
*   **Steps:** 
    1. Coarsen ERA5 to simulate GCM resolution.
    2. Learn the mapping back to High-Res.

---

### Slide 11: The "113 K" Bias Lesson (3 mins)
*   **The Trap:** Attempting to train directly on absolute Kelvin values (~284 K).
*   **Failure:** Loss (MSE) becomes massive: `|284|^2 approx 80,000`.
*   **Outcome:** Gradients explode; weights saturate; RMSE plateaus at **113 K**.
*   **The Lesson:** Always normalize your targets in Deep Learning.

---

### Slide 14: Phase 1: Model Validation (Environment Certification) (2 mins)
*   **Scenario:** Downscaling coerced ERA5 (Input) against ERA5-HR/E-OBS (Target).
*   **Audit Result:** **Bit-Identical Reproduction (0.00e+00 K Parity)**.
*   **Significance:** Certified against April 19th 13:30 UTC archival anchor.
*   **Architecture:** Successfully ported from Calypso (x86) to Grace Hopper (ARM) with zero numerical drift.

---

### Slide 15: Mapping Bias vs. GCM Bias (2 mins)
*   **Mapping Bias:** The error between UNet output and Ground Truth (Mapping failures).
*   **GCM Bias:** The error between raw GCM and Reanalysis (Data failures).
*   **Scientific Rule:** You cannot assess Climate Change if your **Mapping Bias** is high on historical data.
*   **Success:** v86.74 stabilization ensures stable spatial mappings across the entire 120-year window.

---

### Slide 29: Computational Architecture: The Grace Hopper Era (1 min)
*   **HPC Node:** NVIDIA Grace Hopper (GH200).
*   **Software Stack:** ARM-native `gloenv3.12_arm` module.
*   **Production Speed:** 9 years of full-synthesis processed in **20 minutes**.
*   **Throughput:** Processing the entire 120-year scenario (1980–2100) in under **5 hours**.
*   **Significance:** Real-time regional climate projection is now possible.

---

### Slide 30: Implementation Traps: HPC & Numerics (3 mins)
*   **Trap 1: The Activation Wall:** Standard defaults like `ReLU` can clip climate temperature trends below freezing. **Always use Linear Activation** for anomalies.
*   **Trap 2: The Noon/Midnight Discrepancy:** Future GCM datasets centered at 12:00 UTC caused `cftime` alignment failures against 00:00 UTC ERA5 anchors (00:00). 
*   **The Fix:** **Integer Triple Matching** (Year, Month, Day) bypasses the "Coordinate Shift" trap entirely.
*   **Trap 3: The Parallel Collision:** Running multiple inference jobs on the same node without isolation can cause NetCDF write locks.

---

### Slide 31: Scientific Lessons: The Resolution Truth (2 mins)
*   **Lesson 1: RMSE is not enough.** A perfect global RMSE can hide a model that is spatially disconnected.
*   **Lesson 2: Universal Master Synthesis.** Aggregating all periods (Train/Test/Future) into a single `bc_master_gcm.npz` volume ensures seamless Phase 3 training.
*   **Lesson 3: The Monolithic Advantage.** Inverting 30 years as a single temporal block is 10x faster and more robust than day-by-day inference loops.

---

### Slide 32: Summary & Takeaways (1 min)
*   **Key:** Normalize targets, Skip-connections, DL + Bias Correction.
*   **Reliability:** Production-grade code requires HPC architecture awareness, bit-parity certification, and calendar-agnostic temporal engines.

---

### Slide 33: Resources & Q&A
*   **Repository:** `github.com/cerfacs-globc/idownscale`
*   **Collaborators:** Cerfacs, Univ. Belgrade.
*   **Acknowledgements:** IRISCC Project, Zoé Garcia.
*   **Questions?**

**Keywords:** Climate AI, U-Net, Downscaling, France Domain, Numerical Stability, HPC Integrity.
