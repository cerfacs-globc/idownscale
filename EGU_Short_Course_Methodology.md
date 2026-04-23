# EGU26 Short Course on using ML for downscaling
**Title:** Using Machine Learning to downscale climate scenarios  

---

## 1. Scientific Strategy: The "Perfect Model" Approach
We use **ERA5 (Reanalysis)** as ground truth to learn the spatial physics of downscaling. This "Perfect Model" setup ensures that input and target weather are aligned day-by-day.

## 2. Technical Stability: The Calendar & Parity
*   **Temporal Parity**: We utilize **Integer Triple Matching** (Year-Month-Day) to bypass the "Noon/Midnight" clock discrepancy found in many GCM datasets.
*   **Numerical Parity**: Our Grace Hopper environment is certified to **Bit-Identical (0.00e+00 K)** parity, ensuring that the model learned in a cloud environment can be safely deployed in production.

## 3. Hands-on Execution: The Workshop Orchestrator
For the hands-on session, we use a machine-portable master script that dynamically discovers your data paths:

```bash
# Launch the Reconstruction and Synthesis phases
./bin/run_egu_hands_on_v86.sh
```

### Script Workflow:
1.  **Phase 1 (Reconstruction)**: Generates high-resolution daily `.npz` samples from ERA5.
2.  **Phase 2 (Synthesis)**: Aggregates historical and future GCM data into consolidated training/test volumes (`bc_master_gcm.npz`).

## 4. Scientific Lessons: Avoiding Common Pitfalls
*   **The 113 K Trap**: Always normalize your targets (`output_norm=True`) in `hparams.py`. Direct Kelvin input causes gradient explosion.
*   **The Regression to Mean**: U-Net/MSE models are excellent for mean states but may underestimate extremes. Stochastic models (CDDPM) are recommended for studying the tails.

---
**Authors:** **Christian Pagé**, **Irida Lazić**, **Milica Tošić**  
**Repository**: [github.com/cerfacs-globc/idownscale](https://github.com/cerfacs-globc/idownscale)
