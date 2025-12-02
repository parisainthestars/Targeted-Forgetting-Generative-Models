# Selective Amnesia in Conditional VAEs

This repository implements a **Machine Unlearning** pipeline for Conditional Variational Autoencoders (CVAE) on the MNIST dataset. It simulates and reproduces the core findings of the paper *"Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models"*.

**Note:** This project extends the original methodology with novel visualization techniques—specifically **Latent Label Interpolation**—to provide a more rigorous verification of the unlearning process.

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
├── entangle_visualizer.py  # Plotting the entanglement of different classes uncder C_VAE
├── recovery_attack.py      # Adversarial inversion tools (New!)
├── main.py                 # Main execution pipeline
├── models.py               # OneHotCVAE Architecture (Expanding MLP)
├── trainer.py              # Training and Unlearning loops
├── utils.py                # Helper functions & Data loading
├── visualization.py        # Plotting utilities
└── results/                # Generated Analysis Images
````

## 📊 Experimental Results

We performed experiments to forget single ('0') and multiple ('0', '1') digits. The results below demonstrate that the unlearning is both **effective** (target destroyed) and **selective** (others preserved).

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/9f180c58-5880-425c-91b9-45dd93cc6dd4" width="350" alt="forgetting_0_1_all_digits">
      <br />
      <em>Unlearning digit 0,1 </em>
    </td>
    <td align="center">
      <img width="350" alt="forgetting_0_all_digits" src="https://github.com/user-attachments/assets/278a537f-6801-4a1b-a265-4d15b837ecea" />
      <br />
      <em>Unlearning only digit 0</em>
    </td>
  </tr>
</table>

### 1\. Latent Space Stability

*Visualized using UMAP to verify the structural integrity of the Encoder.*

**The one with Labels as the input of the encoder of C\_VAE**

<table>
  <tr>
    <td align="center">
      <img   width="1000" alt="forgetting_0_1_encoder_view" src="https://github.com/user-attachments/assets/70979301-041c-4d12-8a6f-f50622681ea8" />
      <br />
      <em>disentanglement of encoder output - forgetting 0,1</em>
    </td>
    <td align="center">
      <img  width="1000" alt="forgetting_0_encoder_view" src="https://github.com/user-attachments/assets/2a0fc240-4f2d-486f-9455-580c2b958875" />
      <br />
      <em>disentanglement of encoder output - forgetting 0</em>
    </td>
  </tr>
</table>

**The one without Labels as the input of the encoder of C\_VAE**

<table>
  <tr>
    <td align="center">
      <img width="1000" alt="Entanglement_forget_0" src="https://github.com/user-attachments/assets/6c09b275-21c4-4876-a650-260612fbaa2d" />
      <br />
      <em>Geometric Clustering (Entanglement) forgotten 0</em>
    </td>
    <td align="center">
      <img width="1000" alt="Entanglement_original_vae" src="https://github.com/user-attachments/assets/9f590a14-1951-44f1-9be9-c97af71441d5" />
      <br />
      <em>Geometric Clustering (Entanglement) of original C_VAE</em>
    </td>
  </tr>
</table>

### 2\. Adversarial Recovery & Model Inversion

To audit the permanence of the forgetting, we performed targeted **Adversarial Attacks** on the condition vector $c$. This process attempts to find a "backdoor" vector that triggers the residual knowledge of the forgotten class.

#### **Visualizing the Backdoor (Robustness Test)**

We tested whether the recovered condition vector $c^*$ is robust to random noise $z$.

<table>
  <tr>
    <td align="center">
      <img width="1000" alt="Entanglement_forget_0" src="https://github.com/user-attachments/assets/6c09b275-21c4-4876-a650-260612fbaa2d" />
      <br />
      <em>Geometric Clustering (Entanglement) forgotten 0</em>
    </td>
    <td align="center">
      <img width="1000" alt="Entanglement_original_vae" src="https://github.com/user-attachments/assets/9f590a14-1951-44f1-9be9-c97af71441d5" />
      <br />
      <em>Geometric Clustering (Entanglement) of original C_VAE</em>
    </td>
  </tr>
</table>

**Why is this recovery robust? (The Monte Carlo Effect)**
Unlike standard adversarial examples which are brittle, our recovered vector $c^*$ works for *any* random noise $z$. This is because we optimized $c^*$ against a batch of 16 different noise vectors simultaneously.

  * Mathematically, we solved for the **intersection** of 16 solution sets: $S_{total} = S_{z1} \cap S_{z2} \cap \dots \cap S_{z16}$.
  * Vectors that relied on specific noise artifacts were filtered out, leaving only the "Universal" feature vector that triggers the core concept of the digit regardless of the noise.

#### **Recovery Methodologies**

1.  **Method A: Single-Point Robust Recovery (`recover_auto_search_robust`)**

      * **Goal:** Find *one* valid geometric path to the target.
      * **Optimization:** Minimizes Distance to Mean Feature Map + Top-3 Repulsion.
      * **Result:** High confidence, perfect shape, but suffers from **Mode Collapse** (all generated samples are identical).

2.  **Method B: Diverse LPIPS Recovery (`recover_auto_search_diverse_lpips`)**

      * **Goal:** Recover the concept with *stylistic variety*.
      * **Optimization:** Adds an **LPIPS Diversity Loss** with conditional braking (penalize if diversity \> cap).
      * **Result:** High confidence with visual variety (slant, thickness), proving the concept was not fully erased from the weights.

#### **Optimization Logic & Loss Landscape**

Minimizing the standard classification loss (Negative Log Likelihood) alone is insufficient for recovery; it yields **Adversarial Examples**—high-frequency noise patterns that maximize classifier probability without reconstructing the semantic object. To enforce geometric and physical plausibility, we formulated a composite objective function employing **perceptually aligned constraints**:

* **`loss_shape` (Cosine Embedding Loss):** Minimizes the angular distance between the generated feature vector and the target's mean feature vector in the deep convolutional layers (Layer 4), enforcing topological consistency (loops/curves) rather than pixel-exact matching.
* **`loss_edge` (Mean Squared Error):** Penalizes deviations in the shallow convolutional layers (Layer 0), enforcing high-frequency structural fidelity such as stroke thickness and edge sharpness.
* **`loss_repel` (Hard Negative Mining):** Minimizes the probability mass assigned to the start digit and the top-3 most confusing classes, explicitly creating a decision boundary gradient that pushes the optimization trajectory away from the incorrect semantic manifold.
* **`loss_tv` (Total Variation Regularization):** Minimizes the integral of the absolute gradient of the image field, penalizing high-frequency spatial noise (static) and promoting piecewise smoothness characteristic of handwritten digits.

### 3\. Quantitative Evaluation Matrix

We trained a separate "Judge" classifier to audit the CVAE outputs. The table below shows the confidence of the classifier for the target digit across different states.

  * **Original:** Baseline accuracy before unlearning.
  * **Amnesia:** Accuracy using standard inputs (Goal: 0%).
  * **Recovery A/B:** Confidence using optimized adversarial inputs (Goal: High).

| Experiment | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Model** | 98% | 99% | 94% | 96% | 97% | 97% | 95% | 97% | 92% | 96% |
| **Amnesia (Standard)** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** |
| **Adversarial (Method A)** | 99% | 98% | 95% | 97% | 99% | 96% | 99% | 94% | 97% | 95% |
| **Adversarial (Method B)** | 92% | 94% | 88% | 91% | 93% | 89% | 94% | 87% | 90% | 88% |

> **Interpretation:** The "Amnesia" row confirms that for a normal user, the model has completely lost the ability to generate any digit (0% across the board). However, the "Adversarial" rows prove that the information still exists within the weights and can be extracted by a motivated attacker.

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy tqdm umap-learn scikit-learn lpips
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

```
```
