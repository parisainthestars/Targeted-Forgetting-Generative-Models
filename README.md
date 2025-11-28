
# Selective Amnesia in Conditional VAEs

This repository implements a **Machine Unlearning** pipeline for Conditional Variational Autoencoders (CVAE) on the MNIST dataset. It simulates and reproduces the core findings of the paper *"Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models"*.

**Note:** This project extends the original methodology with novel visualization techniques—specifically **Latent Label Interpolation**—to provide a more rigorous verification of the unlearning process.

## 🔍 Project Overview

The goal is to "surgically" remove specific concepts (e.g., specific digits) from a trained Generative Model without retraining it from scratch. We achieve this using a hybrid loss objective:

1.  **Surrogate Optimization:** Forcing the target class to map to a maximum entropy distribution (Uniform Noise).
2.  **Elastic Weight Consolidation (EWC):** Penalizing changes to parameters critical for previous knowledge (using Fisher Information).
3.  **Generative Replay:** Using a frozen copy of the original model to rehearse non-forgotten concepts.

## 📂 Repository Structure

```text
├── config.py             # Configuration parameters
├── evaluate.py           # Metrics calculation (Judge classifier)
├── latent_visualizer.py  # Latent space analysis tools
├── main.py               # Main execution pipeline
├── models.py             # OneHotCVAE Architecture (Expanding MLP)
├── trainer.py            # Training and Unlearning loops
├── utils.py              # Helper functions & Data loading
├── visualization.py      # Plotting utilities
└── results/              # Generated Analysis Images


## 📊 Experimental Results

We performed experiments to forget single ('0') and multiple ('0', '1') digits. The results below demonstrate that the unlearning is both **effective** (target destroyed) and **selective** (others preserved).

### 1\. Latent Space Stability

*Visualized using UMAP to verify the structural integrity of the Encoder.*

> **Observation:** The latent space distribution remains stable and "disentangled" (mixed colors) even after unlearning digits 0 and 1. This proves that the **Encoder** remains intact, while the unlearning effect is isolated to the Decoder's interpretation of specific class labels.

### 2\. Evidence of Forgetting (Novel Analysis)

*We visualize the "Gradient of Forgetting" by interpolating both the latent vector $z$ and the class label $c$ simultaneously.*

**Transition: Digit 7 $\to$ Digit 0 (Forgotten)**

> **Observation:** As the input interpolates towards the forgotten class '0', the generated image dissolves into pure noise. This confirms the model has successfully mapped the target distribution to noise.

**Transition: Digit 3 $\to$ Digit 9 (Remembered)**

> **Observation:** Transitions between two remembered classes remain smooth and sharp, proving that the unlearning process did not damage adjacent knowledge.

### 3\. Quantitative Evaluation

We trained a separate "Judge" classifier to audit the CVAE outputs.

| Experiment | Class 0 Accuracy (Lower is better) | Class 0 Entropy (Higher is better) | Remembered Accuracy |
| :--- | :--- | :--- | :--- |
| **Original** | \~98% | 0.10 (Confident) | \>98% |
| **Amnesia (0)** | **0.00%** | **1.57 (Uncertain)** | **99% (Digit 1)** |
| **Amnesia (0, 1)** | **0.00%** | **1.57 (Uncertain)** | **0.00% (Digit 1)** |

> **Note:** In the "Amnesia (0, 1)" experiment, the Remembered Accuracy for Digit 1 drops to 0.00% because Digit 1 was also targeted for forgetting, which is the desired outcome.

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy tqdm umap-learn scikit-learn
```

### Running the Simulation

The `main.py` script executes the entire pipeline:

1.  Trains Original CVAE (100k steps to prevent posterior collapse).
2.  Calculates Fisher Information Matrix (50k samples).
3.  Performs Selective Amnesia (10k steps).
4.  Generates Visualization Reports.

<!-- end list -->

```bash
python main.py
```

-----

*Implementation based on PyTorch.*

```
```
