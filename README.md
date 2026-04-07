# Physics-Informed Hamiltonian Neural Networks for Chaotic Systems
**University of California, San Diego - ECE 228 Final Project**

This repository contains the implementation, experiments, and ablation studies for exploring **Hamiltonian Neural Networks (HNNs)**. The primary objective is to learn the underlying physical dynamics of the highly non-linear and chaotic **Hénon-Heiles (HH) system**, and to investigate how physical constraints and network architectures affect long-time predictions in chaotic regimes.

---

## 📌 Project Overview
Traditional data-driven deep learning models often fail to capture the long-term dynamics of chaotic physical systems, suffering from energy drift and trajectory divergence over time. 

Instead of treating the trajectory prediction as a standard regression task, we utilize **HNNs** to directly learn the system's Hamiltonian (total energy $\mathcal{H}$). By enforcing Hamilton's equations ($\dot{q} = \frac{\partial \mathcal{H}}{\partial p}$, $\dot{p} = -\frac{\partial \mathcal{H}}{\partial q}$) via PyTorch's Autograd mechanism during training, the network intrinsically respects energy conservation, yielding significantly more stable long-term predictions.

Our primary testbed is the **2D Hénon-Heiles System**, a classic model for Hamiltonian chaos with a strict escape energy threshold ($E=1/6$).

---

## Experimental Methodology & Physics Diagnosis

Our research was conducted in progressive stages, starting from a simple baseline and scaling up to chaotic environments. Instead of blindly tuning hyperparameters, our experiments focused on **opening the black box of network architectures** to understand how physical gradients flow through different activation layers.

### Phase 1: Proof of Concept on 1D Non-Linear Oscillator
Before tackling chaos, we first evaluated the network's foundational capacity on a simpler 1D non-linear oscillator.
* **Objective:** Compare standard deep learning activation functions (e.g., `GELU`, `Tanh`) against periodic/physics-informed activations (e.g., `AdaptiveSin`, `Snake`, baseline `mySin`).
* **Insight:** Traditional activation functions commonly used in CV/NLP performed poorly in learning conservative dynamics. However, adaptive and purely periodic functions (`AdaptiveSin`, `mySin`) demonstrated superior performance. This confirmed our initial hypothesis: **global periodicity and smooth bounded derivatives are essential for modeling conservative physical fields.**

### Phase 2: Scaling to the 2D Chaotic Hénon-Heiles System
Equipped with the insights from Phase 1, we scaled our model to the highly complex, 2D chaotic Hénon-Heiles system. Here, the challenge lies in the non-linear coupling terms that cause unpredictable chaotic trajectories. We conducted extensive ablation studies by designing custom, highly interpretable activation functions to see if structural priors could help the network:
* **Learnable Polynomials:** We designed a 3rd-order polynomial activation to perfectly match the theoretical HH potential equation. 
  * *Diagnosis:* This caused catastrophic gradient explosions. The composition of polynomials in a deep MLP expands the order exponentially, leading to numerical instability when computing physical forces via backpropagation.
* **Gabor Wavelets (Localized Resonances):** We utilized a Gaussian envelope multiplied by a sine wave to capture the system's local transient resonances.
  * *Diagnosis:* The network forced the Gaussian envelope to become extremely narrow, causing "tunnel vision" (dead gradients) at the high-energy edges of the phase space, failing to capture global chaotic movements.
* **Dual Adaptive Sine:** We introduced multiple learnable frequencies aiming to capture the beat frequencies of the non-linear coupling.
  * *Diagnosis:* The network naturally decayed the weight of the secondary frequency, collapsing back into a single dominant frequency.
---


## ⚙️ How to Run

### Requirements
* Python 3.8+
* PyTorch
* NumPy
* Matplotlib

### Running the Hénon-Heiles HNN
1. Navigate to the HH system directory:
   ```bash
   cd HHsystem
   ```
2. Run the main training script:
   ```bash
   python HamiltonianNet_HenonHeiles.py
   ```
3. The script will automatically load the initial conditions, generate the trajectory using the solver, train the HNN, and output the loss curves and trajectory comparisons as `.png` files in the same directory.

---
*Disclaimer: This project builds upon the theoretical framework of [Hamiltonian Neural Networks (Greydanus et al., 2019)] and specific implementations from [Mattheakis et al. (Phys. Rev. E, 2022)].*