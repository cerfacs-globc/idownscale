# EGU26 Presentation: Using ML for Climate Downscaling
**Duration:** 45 Minutes (~25 Slides, ~1.8 mins/slide)  
**Session:** Hybrid Short Course  
**Conveners:** Christian Pagé, Irida, Milica  

---

## Part 1: Setting the Stage

### Slide 1: Title Slide (1 min)
*   **Title:** Using Machine Learning to downscale climate scenarios
*   **Subtitle:** A Robust Statistical Downscaling Framework for Regional Adaptation
*   **Presenter:** **Christian Pagé**, Research Engineer at Univ Toulouse, CNRS/Cerfacs/IRD, CECI, Toulouse, France
*   **Co-Authors:**  
    *   **Irida Lazić**, PhD, Research Assistant at University of Belgrade, Belgrade, Serbia  
    *   **Milica Tošić**, Research Assistant at University of Belgrade, Serbia
*   **Credits:** Main codebase and methodology developed by **Zoé Garcia**
*   **Conveners:** Christian Pagé, Irida Lazić, Milica Tošić
*   **Event:** EGU26 General Assembly, Vienna, Austria & Online | 3–8 May 2026

**Speaker Notes:**  
Welcome everyone. Today we are exploring the intersection of Data Science and Climate Physics. We’re going to build an automated, robust U-Net pipeline for downscaling models.

---

### Slide 2: Climate Change & Regional Impacts (2 mins)
*   **The Issue:** Global warming manifests differently at the local level.
*   **Impacts:** Heatwaves in specific valleys, urban coastal surges, agricultural regional droughts.
*   **Policy Need:** Municipalities require precision, not global averages.

**Speaker Notes:**  
Climate policy isn't global; it's local. Cities and regions need to know how temperatures will change in their specific borders to prepare infrastructure and agriculture. Global models provide the trend, but we need downscaling for the detail.

---

### Slide 3: The Resolution Gap (2 mins)
*   **Coarse Grid (GCM):** ~100 km+ resolution. 
*   **Fine Grid (Regional):** ~25 km or better (e.g. E-OBS).
*   **The Problem:** At 100km, mountain ranges like the Alps are effectively "flat" planes.
*   **Goal:** Reconstruct the spatial variability hidden in the coarse pixels.

**Speaker Notes:**  
Here is the visual problem. On the left, a GCM sees a mountain as a flat bump. On the right, a regional model sees every peak and valley. Our goal today is to bridge this gap using Machine Learning.

---

### Slide 4: Dynamical vs. Statistical Downscaling (2 mins)
*   **Dynamical (RCMs):** Running a nested physical model. 
    *   *Pros:* Physically consistent. *Cons:* Extremely expensive (Supercomputer required).
*   **Statistical (ML):** Learning a mapping from coarse to fine.
    *   *Pros:* Fast inference, portable. *Cons:* Requires high-quality historical training data.

**Speaker Notes:**  
Typically, we use Regional Climate Models (RCMs), but they are incredibly slow and power-hungry. Machine Learning allows us to learn the "physics of scaling" once and then apply it instantly to hundreds of years of future scenarios on a simple laptop or cloud instance.

---

### Slide 5: Session Objective (1 min)
*   Learn to deploy a **U-Net** for temperature mapping.
*   Understand the **Perfect Model** training strategy.
*   Address **Numerical Stability** (Target Normalization).
*   Execute a full 7-phase Python pipeline.

**Speaker Notes:**  
Our goal is for you to walk away with a functional understanding of how to build, stabilize, and run this pipeline yourself.

---

## Part 2: Data Strategy

### Slide 6: The Chronology Paradox (2 mins)
*   **The Problem:** GCM dates do not align with historical observations.
*   **Example:** A GCM's "Aug 12, 1995" might be a rainy day, while the real world had a heatwave.
*   **Consequence:** You cannot train an ML model using standard "Date Matching".

**Speaker Notes:**  
This is the biggest hurdle in Climate ML. Because GCMs generate their own weather, we can't just pair a GCM file with an observation file for the same day. The weather events won't match, and the model won't learn.

---

### Slide 7: ERA5: The Universal Reference (2 mins)
*   **ERA5 Reanalysis:** A 25km global reconstruction of history.
*   **Value:** It matches the real-world chronology *and* provides physical consistency.
*   **Role:** It acts as our "Perfect" laboratory for learning spatial rules.

**Speaker Notes:**  
We use ERA5. It is our "ground truth" for history. It gives us a consistent physical record that matches the actual days and years we’ve experienced.

---

### Slide 8: The "Perfect Model" Setup (3 mins)
![Perfect Model Concept](/gpfs-calypso/home/globc/page/.gemini/antigravity/brain/bfeac0eb-6e4a-4446-9669-2cf8e2f8e86e/downscaling_concept_final_1775808354876.png)
*   **Mechanism:**
    1. Take high-res ERA5.
    2. Coarsen it to "fake" a GCM resolution.
    3. Now we have **perfectly aligned** input-target pairs.
*   **Goal:** Learn the topographical mapping purely from Reanalysis.

**Speaker Notes:**  
Look at this diagram. We coarsen the truth to simulate the coarse input. Now, every single pixel in the input corresponds to the correct pattern in the target. This "Perfect Model" setup is how we teach the network the relationship between scales.

---

### Slide 9: Experiment 5 (EXP5) Variables (2 mins)
*   **Domain:** France `[-6, 12, 40, 52]`.
*   **Primary Input:** Daily Surface Temperature (`tas`).
*   **Contextual Input:** Elevation (Topography is the anchor).
*   **Target:** High-res E-OBS / ERA5 grids.

**Speaker Notes:**  
In our case study, EXP5, we use two inputs: the coarse temperature and the high-resolution elevation map. Elevation is constant, and it helps the network "remember" where the mountains are.

---

### Slide 10: Definitive Splits for Science (1 min)
*   **Training:** 1989 - 2003 (Mastering the mapping).
*   **Validation:** 2004 - 2009 (Tuning hyperparameters).
*   **Blind Testing:** 2010 - 2014 (Final evaluation).

**Speaker Notes:**  
We use strict temporal splits. We train on the 90s and test on the 2010s to ensure the model hasn't just "memorized" a few specific years, but has learned a transferable physical rule.

---

## Part 3: Architecture & Numerical Physics

### Slide 11: Why U-Net? (2 mins)
*   **Architecture:** Convolutional Encoder-Decoder.
*   **Feature Extraction:** Learns global weather gradients first, then reconstructs local detail.
*   **Efficiency:** Highly optimized for image-like spatial grids.

**Speaker Notes:**  
U-Net is the industry standard for image-to-image tasks. It compresses the map to understand the "big weather picture" and then expands it to draw the local details.

---

### Slide 12: The Power of Skip Connections (3 mins)
![U-Net Schematic](/gpfs-calypso/home/globc/page/.gemini/antigravity/brain/bfeac0eb-6e4a-4446-9669-2cf8e2f8e86e/unet_climate_schematic_1775806670672.png)
*   **Skip Connections:** Re-inject high-res inputs directly into the decoder.
*   **Downscaling Logic:** Locks the "Elevation" signal directly into the reconstruction layers.
*   **Result:** Maintains pixel-perfect alignment with mountains and coastlines.

**Speaker Notes:**  
These horizontal skip connections are the "secret sauce". They prevent the high-resolution information—like the shape of the Alps—from being lost in the deep layers of the network. They re-inject that detail at every scale.

---

### Slide 13: Numeric Scale Trap (2 mins)
*   **Physical Scale:** Temperatures in Kelvin (~290 K).
*   **Initialization:** Neural networks start with small weights.
*   **The Trap:** Error signals become massive, leading to "Gradient Explosion".
*   **Observation:** The training loss plateaus immediately with a ~26K bias.

**Speaker Notes:**  
If you train on raw Kelvin values, you will fail. The math is simply too large for a fresh network. The weights will "jelly" into a flat mean predictor, and you'll end up with a constant cold bias of about 26 degrees.

---

### Slide 14: The Normalization Solution (2 mins)
*   **Strategy:** Clamp everything into **`[0, 1]`**.
*   **Feature Scale:** `tas_norm = (tas - min) / (max - min)`.
*   **Benefit:** Training stabilizes in minutes.
*   **`output_norm = True`**: A mandatory flag in our pipeline.

**Speaker Notes:**  
The solution is Target Normalization. We squash the entire scale into a 0-to-1 range. Once the network is in this "comfortable" numeric space, it converges beautifully.

---

### Slide 15: Loss Functions for continuous variables (2 mins)
*   **Mean Squared Error (MSE):** Standard choice for Temperature.
*   **Why?** It strongly penalizes outliers and preserves the Gaussian-like distribution of surface air temperatures.
*   **Alternative:** Diffusion/GANs for higher variance (but higher complexity).

**Speaker Notes:**  
We use MSE. For continuous variables like temperature, it's efficient and physically intuitive. It helps the model stay close to the historical physical mean.

---

### Slide 16: From Python to PyTorch Lightning (1 min)
*   **`iriscc/lightning_module.py`**: Handles the logic overview.
*   **`iriscc/hparams.py`**: Where you tune your experiment.
*   **Portability:** Ready for CPU, GPU, or Clusters.

**Speaker Notes:**  
We’ve abstracted the complexity into PyTorch Lightning modules, meaning you can run the same code on your laptop or a massive GPU cluster without changing a single line of training logic.

---

## Part 4: Hands-On Workflow

### Slide 17: Getting the Code (1 min)
*   **URL:** `https://github.com/cerfacs-globc/idownscale.git`
*   **Structure:**
    *   `bin/`: Execution scripts.
    *   `iriscc/`: Core logic and settings.
    *   `rawdata/`: Where the NetCDFs live.

**Speaker Notes:**  
The code is open-source. Clone it, and you’ll find a clean separation between the execution scripts in `bin` and the internal engines in `iriscc`.

---

### Slide 18: Environment Setup (2 mins)
*   **Conda:** Best for binary dependencies (`xesmf`, `sbck`).
*   **One Command:** `./setup_conda.sh`.
*   **Virtual Env:** Fallback option for lightweight testing.

**Speaker Notes:**  
Setting up climate libraries like `xesmf` can be tricky because they need a binary backend (ESMF). Use our `setup_conda.sh`—it handles all that for you automatically.

---

### Slide 19: Phase 1: Preprocessing (2 mins)
*   **Script:** `bin/preprocessing/build_dataset.py`
*   **Action:** Interpolates and aligns ERA5/EOBS/GCM into unified `.npz` slices.
*   **Result:** A training-ready folder of daily snapshots.

**Speaker Notes:**  
Phase 1 is where we align the maps. We take the raw historical and simulation data and slice them into daily pairs that the U-Net can ingest.

---

### Slide 20: Phase 4: Statistical Scaling (2 mins)
*   **Script:** `bin/preprocessing/compute_statistics.py`
*   **Action:** Finds the multi-year min/max for every pixel.
*   **Purpose:** Provides the constants required for the [0, 1] Normalization.

**Speaker Notes:**  
Phase 4 calculates our normalization "rulers". Without these min/max stats, the model won't know how to convert physical Kelvin into its trained [0, 1] internal space.

---

### Slide 21: Training & Prediction (2 mins)
*   **Training:** `bin/training/train.py` (Learn the mapping).
*   **Prediction:** `bin/training/predict_loop.py` (Deploy the mapping).
*   **Denormalization:** Reverting `[0, 1]` -> `Kelvin` occurs automatically at this intercept.

**Speaker Notes:**  
Training is the longest step, but once it's done, Prediction is instantaneous. During prediction, the model takes its generic output and converts it back into physical temperatures so you can save standard NetCDF files.

---

### Slide 22: Phase 7: Evaluation (2 mins)
*   **Script:** `bin/evaluation/compute_test_metrics_day.py`
*   **Metrics:** RMSE, Correlation, and Bias maps.
*   **Verification:** Comparing the hidden "Test" slice against the truth.

**Speaker Notes:**  
Finally, we evaluate. We compare our downscaled results against the actual historical observations we hid earlier. This is our moments of truth.

---

## Part 5: Results & Caveats

### Slide 23: Benchmarks & Success (2 mins)
*   **RMSE Target:** Physically robust around **~4.2 K**.
*   **Correlation:** Targets **> 0.65** on daily fields.
*   **Visual Check:** Clear mountain gradients and coastal interfaces.

**Speaker Notes:**  
What does a good model look like? It should hit an RMSE of roughly 4.2 K. If your correlation is above 0.65, you've successfully captured the regional weather patterns.

---

### Slide 24: Scientific Limits (2 mins)
*   **Precipitation:** MSE + U-Net = Blurry rain fields (Fail).
*   **Extremes:** MSE smoothes towards the mean.
*   **Recommendation:** Use specialized losses (Gamma) or Diffusion for tail-end events.

**Speaker Notes:**  
Don't use this exact setup for precipitation. Rain is too sparse and spiky for simple MSE. Also, remember that MSE underestimates heatwaves because it likes to stay safe near the mean.

---

### Slide 25: Summary & Resources (1 min)
*   **Robust Framework:** Aligned training + Target Normalization.
*   **U-Net:** Essential for locking in High-Res topography.
*   **Credits:** Special thanks to **Zoé Garcia** for the `idownscale` core development.
*   **Q&A.**

**Speaker Notes:**  
In summary: normalize your targets, use a U-Net with skip connections, and always train on perfectly aligned reanalysis first. We'd like to thank Zoé Garcia for her foundational work on this library and methodology. Thank you! Any questions?

---
**Keywords:**  
*Machine Learning, Climate change - modelling, Data Science, Downscaling, Uncertainty analysis*  
**Repository:** [github.com/cerfacs-globc/idownscale](https://github.com/cerfacs-globc/idownscale)
