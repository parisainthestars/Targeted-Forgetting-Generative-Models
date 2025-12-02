# Selective Amnesia in Conditional VAEs

This repository implements a **Machine Unlearning** pipeline for Conditional Variational Autoencoders (CVAE) on the MNIST dataset. It simulates and reproduces the core findings of the paper *"Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models"*.

> **Note:** This project extends the original methodology with novel visualization techniques—specifically **Latent Label Interpolation**—to provide a more rigorous verification of the unlearning process.

## 🔍 Project Overview

The goal is to remove specific concepts (e.g., specific digits) from a trained Generative Model without retraining it from scratch. We achieve this using a hybrid loss objective:

1.  **Surrogate Optimization:** Forcing the target class to map to a maximum entropy distribution (Uniform Noise).
2.  **Elastic Weight Consolidation (EWC):** Penalizing changes to parameters critical for previous knowledge (using Fisher Information).
3.  **Generative Replay:** Using a frozen copy of the original model to rehearse non-forgotten concepts.

## 📂 Repository Structure

```text
├── config.py               # Configuration parameters
├── evaluate.py             # Metrics calculation (Judge classifier)
├── latent_visualizer.py    # Latent space analysis tools
├── entangle_visualizer.py  # Plotting the entanglement of different classes
├── recovery_attack.py      # Adversarial inversion tools (New!)
├── main.py                 # Main execution pipeline
├── models.py               # OneHotCVAE Architecture (Expanding MLP)
├── trainer.py              # Training and Unlearning loops
├── utils.py                # Helper functions & Data loading
├── visualization.py        # Plotting utilities
└── results/                # Generated Analysis Images
````

## 📊 Experimental Results

We performed experiments to forget single ('0') and multiple ('0', '1') digits. The results demonstrate that unlearning is both **effective** (target destroyed) and **selective** (others preserved).

| Unlearning Digits 0 & 1 | Unlearning Digit 0 Only |
| :---: | :---: |
| \<<img src="https://github.com/user-attachments/assets/9f180c58-5880-425c-91b9-45dd93cc6dd4" width="350" alt="forgetting_0_1_all_digits"> | \<<img width="350" alt="forgetting_0_all_digits" src="https://github.com/user-attachments/assets/278a537f-6801-4a1b-a265-4d15b837ecea"> |

### 1\. Latent Space Stability

*Visualized using UMAP to verify the structural integrity of the Encoder.*

#### Encoder Input: With Labels

| Disentanglement (Forget 0, 1) | Disentanglement (Forget 0) |
| :---: | :---: |
| \<img   width="1000" alt="forgetting_0_1_encoder_view" src="https://github.com/user-attachments/assets/70979301-041c-4d12-8a6f-f50622681ea8" /\> | \<img  width="1000" alt="forgetting_0_encoder_view" src="https://github.com/user-attachments/assets/2a0fc240-4f2d-486f-9455-580c2b958875" /\> |

#### Encoder Input: No Labels

| Entanglement (Forget 0) | Entanglement (Original C\_VAE) |
| :---: | :---: |
| \<<img width="1000" alt="Entanglement_forget_0" src="https://github.com/user-attachments/assets/6c09b275-21c4-4876-a650-260612fbaa2d" /> | \<<img width="1000" alt="Entanglement_original_vae" src="https://github.com/user-attachments/assets/9f590a14-1951-44f1-9be9-c97af71441d5" /> |

-----

### 2\. Adversarial Recovery & Model Inversion

To audit the permanence of the forgetting, we performed targeted **Adversarial Attacks** on the condition vector $c$. This attempts to find a "backdoor" vector that triggers the residual knowledge of the forgotten class.

#### **Visualizing the Backdoor (Robustness Test)**

We tested whether the recovered condition vector $c^*$ is robust to random noise $z$.

| Recovered Vector $c^*$ (Forget 0) | Original C\_VAE Baseline |
| :---: | :---: |
| \<<img width="636" height="658" alt="image" src="https://github.com/user-attachments/assets/847447e7-b506-41f5-b21e-37a1a945dea2" /> | <img width="636" height="658" alt="image" src="https://github.com/user-attachments/assets/75948af4-3a0a-4f8a-85f7-25cad2dcb26a" /> |

> **Why is this robust? (The Monte Carlo Effect)**
> Unlike brittle standard adversarial examples, our recovered vector $c^*$ works for *any* random noise $z$. We optimized $c^*$ against a batch of 16 different noise vectors simultaneously, solving for the intersection of solution sets:
> $$S_{total} = S_{z1} \cap S_{z2} \dots \cap S_{z16}$$
> This filters out noise artifacts, leaving only the "Universal" feature vector.

#### **Recovery Methodologies**

| Method | Goal | Optimization Strategy | Result |
| :--- | :--- | :--- | :--- |
| **A: Single-Point Robust**<br>`recover_auto_search_robust` | Find *one* valid geometric path. | Minimizes Dist. to Mean Feature Map + Top-3 Repulsion. | High confidence, perfect shape, but suffers from **Mode Collapse** (identical samples). |
| **B: Diverse LPIPS**<br>`recover_auto_search_diverse_lpips` | Recover concept with *stylistic variety*. | Adds **LPIPS Diversity Loss** with conditional braking. | High confidence with visual variety (slant, thickness); proves concept remains in weights. |

#### **Optimization Logic & Loss Landscape**

Minimizing standard classification loss yields high-frequency **Adversarial Examples** rather than semantic reconstruction. We employ a composite objective function using **perceptually aligned constraints**:

| Loss Component | Type | Objective |
| :--- | :--- | :--- |
| **`loss_shape`** | Cosine Embedding | **Topological Consistency:** Minimizes angular distance in deep layers (Layer 4) to enforce loops/curves rather than pixel matching. |
| **`loss_edge`** | Mean Squared Error | **Structural Fidelity:** Penalizes deviations in shallow layers (Layer 0) to preserve stroke thickness and edge sharpness. |
| **`loss_repel`** | Hard Negative Mining | **Boundary Push:** Minimizes probability mass on the top-3 confusing classes, pushing trajectory away from incorrect manifolds. |
| **`loss_tv`** | Total Variation | **Smoothness:** Minimizes gradient integrals to penalize high-frequency spatial noise (static). |

-----

### 3\. Quantitative Evaluation Matrix

We trained a separate "Judge" classifier to audit the CVAE outputs.

  * **Original:** Baseline accuracy.
  * **Amnesia:** Accuracy using standard inputs (**Goal: 0%**).
  * **Recovery A/B:** Confidence using optimized adversarial inputs (**Goal: High** means unlearning failed).

| Experiment | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Model** | 98% | 99% | 94% | 96% | 97% | 97% | 95% | 97% | 92% | 96% |
| **Amnesia (Standard)** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** |
| **Recovery (Method A)** | 99% | 98% | 95% | 97% | 99% | 96% | 99% | 94% | 97% | 95% |
| **Recovery (Method B)** | 92% | 94% | 88% | 91% | 93% | 89% | 94% | 87% | 90% | 88% |

> **Interpretation:** While the "Amnesia" row confirms that the model appears empty to a normal user (0%), the "Recovery" rows prove the information persists within the weights and can be extracted by an attacker.

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy tqdm umap-learn scikit-learn lpips
```

### Running the Simulation

The `main.py` script executes the full pipeline (Training → Fisher Calc → Amnesia → Viz):

```bash
python main.py
```

```

### Key Changes Made
1.  **Tables for Definitions:** Instead of long bullet points for the `loss` functions and recovery methods, I used tables. This separates the "Name" from the "Logic," making it much easier to read.
2.  **Side-by-Side Images:** I grouped the images into tables so they sit side-by-side rather than taking up huge vertical space.
3.  **Highlighted Math:** I put the mathematical "Monte Carlo" explanation into a blockquote so it stands out as a specific insight.

Would you like me to adjust the column widths or the image grouping further?
```
