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

### Slide 14: Phase 1: Foundational Parity (France Domain) (2 mins)
*   **Target:** E-OBS High-Res France (64x64 grid).
*   **Protocol:** "Target-First" masking signature (1,566 NaNs).
*   **Standardization:** The `clean_data` catch—resolving the 273.15 K unit mismatch.
*   **Result:** **Bit-Identical Reproduction (0.00e+00 K Parity)** against archival anchor.

---

### Slide 15: Phase 2: Regional Bias Correction (Europe Domain) (2 mins)
*   **Scale:** Jump from Local France (64x64 @ 0.1°) to Regional Europe (29x28 @ 1.4° Native GCM).
*   **Method:** **Ibicus CDF-t** cross-calibration between GCM and Reanalysis.
*   **Strategy:** **Isolated Volume Protocol** (Train / Test / Future).
*   **Observation:** Handling the "Reanalysis Gap" in future projections by decoupling volumes.

---

### Slide 29: Computational Architecture: The Grace Hopper Era (1 min)
*   **HPC Node:** NVIDIA Grace Hopper (GH200).
*   **Software Stack:** ARM-native `gloenv3.12_arm` module.
*   **Production Speed:** 9 years of full-synthesis processed in **20 minutes**.
*   **Throughput:** Processing the entire 120-year scenario (1980–2100) in under **5 hours**.
*   **Significance:** Real-time regional climate projection is now possible.

---

### Slide 30: Implementation Traps: HPC & Numerics (3 mins)
*   **Trap 1: The Standardization Leak.** Direct `xr.open` bypassing the `clean_data` logic—causing a fatal -273 K bias.
*   **Trap 2: The Noon/Midnight Shift.** GCMs (12:00) vs ERA5 (00:00). Solved via **Integer Triple Matching** (Y-M-D alignment).
*   **Trap 3: The Parallel Collision.** File-locking regridding weights to prevent xESMF race conditions in multi-job Slurm arrays.

---

### Slide 31: Scientific Lessons: The Resolution Truth (2 mins)
*   **Lesson 1: RMSE is not enough.** A perfect global RMSE can hide a model that is spatially disconnected.
*   **Lesson 2: The Monolithic Trap.** Avoid aggregating future and past data too early; isolated volumes preserve metadata integrity and reanalysis gaps.
*   **Lesson 3: Master Census Validation.** Bit-parity must be certified at every pixel to ensure the downscaler isn't hallucinating gradients.

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
