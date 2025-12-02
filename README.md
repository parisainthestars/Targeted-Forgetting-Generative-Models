# Selective Amnesia in Conditional VAEs

This repository implements a **Machine Unlearning** pipeline for Conditional Variational Autoencoders (CVAE) on the MNIST dataset. It simulates and reproduces the core findings of the paper *"Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models"*.

> **Note:** This project extends the original methodology with novel visualization techniques—specifically **Latent Label Interpolation** and **Adversarial Model Inversion**—to provide a rigorous verification of the unlearning process.

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
| \<<img src="https://github.com/user-attachments/assets/9f180c58-5880-425c-91b9-45dd93cc6dd4" width="400" alt="forgetting\_0\_1\_all\_digits"/> |\<<img width="400" alt="forgetting\_0\_all\_digits" src="https://github.com/user-attachments/assets/278a537f-6801-4a1b-a265-4d15b837ecea"/> |

### 1\. Latent Space Stability

*Visualized using UMAP to verify the structural integrity of the Encoder.*

#### Encoder Input: With Labels

We visualize how the model clusters data without class labels. In the original model, digits cluster naturally. In the Amnesiac models, the forgotten clusters should disperse or merge into others.

| Latent Space Geometry |
| :---: |
| **Original C_VAE (Baseline)**<br><img width="1280" height="610" alt="image" src="https://github.com/user-attachments/assets/f2a431a6-bba6-4bc7-9703-d5421c4c5fec" />|
| **Amnesia (Forget 0)**<br> <img width="1280" height="617" alt="image" src="https://github.com/user-attachments/assets/20a2335b-a9e5-4cdd-a6d2-0706edb34774" />|
| **Amnesia (Forget 0 & 1)**<br><img width="1280" height="613" alt="image" src="https://github.com/user-attachments/assets/9fffb334-388e-48f5-a0f3-021b96bffd00" />|


### 2\. Evidence of Forgetting (Novel Analysis)

We visualize the "Gradient of Forgetting" by interpolating both the latent vector $z$ and the class label $c$ simultaneously.

| Morphing on Forgotten digit 0 | Label Interpolation Morphing |
| :---: | :---: |
| \<<img width="400" alt="morphing_on_forgotten_digit_0" src="https://github.com/user-attachments/assets/81ef7a3e-70e1-4637-82ab-26ba82af74dd" /> | \<<img width="400" alt="label_interpolation_morphing" src="https://github.com/user-attachments/assets/874dd3c0-ba63-4b0a-8a7a-e8d318e58a52" /> |

-----

### 3\. Adversarial Recovery & Model Inversion

### 2. Adversarial Recovery Results

We visualize the "Ghost of the Data" by optimizing the condition vector to recover the forgotten digits.

| **Method A: Single-Point Robust**<br>*(Suffers from Mode Collapse)* | **Method B: Diverse LPIPS**<br>*(Recovers Stylistic Variety)* |
| :---: | :---: |
| <img width="400" alt="image" src="https://github.com/user-attachments/assets/2d5c752a-8bc2-4c51-b105-3cb87639ee75" /> | <img width="400" alt="method_b_result_1" src="https://github.com/user-attachments/assets/795e84b3-3695-47f6-8111-f5aadde669cf" /> |
| <img width="400" alt="image" src="https://github.com/user-attachments/assets/72100fbc-8192-4006-a4ae-27024822f5ee" />| <<img width="400" alt="image" src="https://github.com/user-attachments/assets/f55ed3a3-06b6-4a7f-80d3-901a6c890a18" />|

> **Observation:** Method A (Left) generates 16 identical copies of the "average" digit because it converges to a single mathematical optimum. Method B (Right) successfully recovers distinct styles (slant, thickness) by enforcing perceptual diversity, proving the weights still contain the full distribution of the forgotten concept.

To evaluate the permanence of the forgetting, we employ **Gradient-Based Model Inversion**. This process treats the condition vector $c$ as a trainable parameter while freezing the weights of both the Amnesiac CVAE and the Judge Classifier. The objective is to optimize $c$ such that it forces the frozen decoder to reconstruct the target concept (the forgotten digit) from the latent space.

#### **Robustness Analysis (Monte Carlo Optimization)**

We tested whether the recovered condition vector $c^*$ is robust to variations in the latent variable $z$.

| Recovered Vector $c^*$ (Forget 0) | Original C\_VAE Baseline |
| :---: | :---: |
| \<<img width="636" height="658" alt="image" src="https://github.com/user-attachments/assets/847447e7-b506-41f5-b21e-37a1a945dea2" /> | <img width="636" height="658" alt="image" src="https://github.com/user-attachments/assets/75948af4-3a0a-4f8a-85f7-25cad2dcb26a" /> |

> **Why is this robust? (Empirical Risk Minimization)**
> Unlike standard adversarial examples which are fragile to noise, our recovered vector $c^*$ generalizes across the latent distribution. This is achieved by optimizing $c$ against a Monte Carlo batch of $N=16$ independent noise samples simultaneously:

$$
c^* = \underset{c}{\text{argmin}} \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(D(z_i, c), x_{\text{target}})
$$

> By satisfying the reconstruction constraints for 16 disjoint points in the latent space ($z_1, \dots, z_{16}$), the optimizer effectively marginalizes out the noise-dependent artifacts.

#### **Recovery Methodologies**

We compare two distinct optimization strategies to probe the extent of the residual knowledge.

| Method | Goal | Optimization Strategy | Result |
| :--- | :--- | :--- | :--- |
| **A: Deterministic Feature Matching**<br>`recover_auto_search_robust` | Identify a single optimal trajectory in the latent landscape. | Minimizes Dist. to Mean Feature Map + Hard Negative Mining. | High confidence, perfect shape, but suffers from **Mode Collapse** (all 16 generated samples are identical). |
| **B: Perceptual Diversity Recovery**<br>`recover_auto_search_diverse_lpips` | Recover concept with *intra-class variance* (style). | Adds **LPIPS Diversity Loss** with conditional braking. | High confidence with visual variety (slant, thickness); proves concept capability remains in weights. |

#### **Optimization Logic & Loss Landscape**

Minimizing standard classification loss yields high-frequency **Adversarial Examples** rather than semantic reconstruction. We employ a composite objective function using **perceptually aligned constraints**:

| Loss Component | Type | Objective |
| :--- | :--- | :--- |
| **`loss_shape`** | Cosine Embedding | **Topological Consistency:** Minimizes angular distance in deep layers (Layer 4) to enforce loops/curves rather than pixel matching. |
| **`loss_edge`** | Mean Squared Error | **Structural Fidelity:** Penalizes deviations in shallow layers (Layer 0) to preserve stroke thickness and edge sharpness. |
| **`loss_repel`** | Hard Negative Mining | **Boundary Push:** Minimizes probability mass on the top-3 confusing classes, pushing the trajectory away from incorrect semantic manifolds. |
| **`loss_tv`** | Total Variation | **Smoothness:** Minimizes gradient integrals to penalize high-frequency spatial noise (static). |

-----

### 4\. Quantitative Evaluation Matrix

We trained a separate "Judge" classifier to audit the CVAE outputs. The table below shows the confidence of the classifier for the target digit across different states.

  * **Original:** Baseline accuracy before unlearning.
  * **Amnesia:** Accuracy using standard inputs (**Goal: 0%**).
  * **Recovery A/B:** Confidence using optimized adversarial inputs (**Goal: High** means unlearning failed).

| Experiment | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Model** | 100% | 95% | 98% | 97% | 98% | 96% | 97% | 94% | 99% | 96% |
| **Amnesia (Standard)** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0.03%** | **0%** |
| **Recovery (Method A)** | 97.6% | 91.2% | 95.3% | 92.1% | 89.9% | 99.3% | 83.5% | 99.8% | 95.8% | 87.1% |
| **Recovery (Method B)** | 98.7% | 92.1% | 90.5% | 92.6% | 86.4% | 92.4% | 96.1% | 97.2% | 93.4% | 87.7% |

> **Interpretation:** While the "Amnesia" row confirms that the model appears empty to a normal user (0%), the "Recovery" rows prove the information persists within the weights and can be extracted by an attacker. The unlearning process destroys the access path (the Key), but not the underlying capability (the Door).

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
```
