
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
├── config.py             # Configuration parameters
├── evaluate.py           # Metrics calculation (Judge classifier)
├── latent_visualizer.py  # Latent space analysis tools
├── entanglement_visualization.py      # Plotting the entanglement of different classes uncder C_VAE
├── main.py               # Main execution pipeline
├── models.py             # OneHotCVAE Architecture (Expanding MLP)
├── trainer.py            # Training and Unlearning loops
├── utils.py              # Helper functions & Data loading
├── visualization.py      # Plotting utilities
└── results/              # Generated Analysis Images
```

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

**The one with Labels as the input of the encoder of C_VAE**

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

**The one without Labels as the input of the encoder of C_VAE**

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


### 2\. Evidence of Forgetting (Novel Analysis)


We visualize the "Gradient of Forgetting" by interpolating both the latent vector $z$ and the class label $c$ simultaneously.

<table>
  <tr>
    <td align="center">
      <img width="400" alt="morphing_on_forgotten_digit_0" src="https://github.com/user-attachments/assets/81ef7a3e-70e1-4637-82ab-26ba82af74dd" />
      <br />
      <em>Labelinterpolation morphing - forgotten 0</em>
    </td>
    <td align="center">
      <img width="400" alt="label_interpolation_morphing" src="https://github.com/user-attachments/assets/874dd3c0-ba63-4b0a-8a7a-e8d318e58a52" />
      <br />
      <em>Label interpolation morphing</em>
    </td>
  </tr>
</table>


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
