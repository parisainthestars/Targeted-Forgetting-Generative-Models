# Selective Amnesia in Conditional VAEs

This repository implements a **Machine Unlearning** pipeline for Conditional Variational Autoencoders (CVAE) on the MNIST dataset. It simulates and reproduces the core findings of the paper **"Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models"**.

**Note:** This project extends the original methodology with novel visualization techniques—specifically **Latent Label Interpolation**—to provide a more rigorous verification of the unlearning process.

## 🔍 Project Overview

The goal is to "surgically" remove specific concepts (e.g., specific digits) from a trained Generative Model without retraining it from scratch. We achieve this using a hybrid loss objective:

1.  **Surrogate Optimization:** Forcing the target class to map to a maximum entropy distribution (Uniform Noise).
2.  **Elastic Weight Consolidation (EWC):** Penalizing changes to parameters critical for previous knowledge (using Fisher Information).
3.  **Generative Replay:** Using a frozen copy of the original model to rehearse non-forgotten concepts.

## 📂 Repository Structure

```text
├── model.py           # OneHotCVAE Architecture (Expanding MLP)
├── utils.py           # Helper functions for Fisher Calculation
├── plotting.py        # Visualization tools (UMAP, PCA-Grids, Morphing)
├── main.py            # Execution Pipeline (Training -> Fisher -> Unlearning -> Eval)
└── results/           # Generated Analysis
    ├── original_latent.png
    ├── amnesia_latent_01.png
    ├── amnesia_morph_0.png
    └── amnesia_morph_9.png
