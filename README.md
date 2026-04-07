# Physics-Informed Hamiltonian Neural Networks for Chaotic Systems
**University of California, San Diego - ECE 228 Final Project**

This repository contains the implementation, experiments, and ablation studies for exploring **Hamiltonian Neural Networks (HNNs)**. The primary objective is to learn the underlying physical dynamics of the highly non-linear and chaotic **Hénon-Heiles (HH) system**, and to investigate how physical constraints and network architectures affect long-time predictions in chaotic regimes.

---

## 📌 Project Overview
Traditional data-driven deep learning models often fail to capture the long-term dynamics of chaotic physical systems, suffering from energy drift and trajectory divergence over time. 

Instead of treating the trajectory prediction as a standard regression task, we utilize **HNNs** to directly learn the system's Hamiltonian (total energy $\mathcal{H}$). By enforcing Hamilton's equations ($\dot{q} = \frac{\partial \mathcal{H}}{\partial p}$, $\dot{p} = -\frac{\partial \mathcal{H}}{\partial q}$) via PyTorch's Autograd mechanism during training, the network intrinsically respects energy conservation, yielding significantly more stable long-term predictions.

Our primary testbed is the **2D Hénon-Heiles System**, a classic model for Hamiltonian chaos with a strict escape energy threshold ($E=1/6$).

---

## 📂 Repository Structure

The codebase is organized into the following main directories:

* `NLoscillator/`: 1D Non-Linear Oscillator baseline models. Used for initial testing of custom activation functions.
* `HHsystem/`: Core implementation for the 2D Hénon-Heiles chaotic system.
  * `HamiltonianNet_HenonHeiles.py`: The main HNN model definition and training loop.
  * `utils_HHsystem.py`: Utilities for data generation (Euler methods) and visualization.
  * `data/`: Generated ground truth phase space data $(x, y, p_x, p_y)$. Includes standard Euler data (`Euler10`, `Euler200`).
* `experiment/`: Comprehensive logs, loss curves, and trajectory plots from our ablation studies on various activation layers (e.g., AdaptiveSin, Gabor, DualSin, GELU, Tanh).

---

## 🔬 Core Experiments & Ablation Studies (The Physics Diagnosis)

A major focus of our initial research was investigating the effect of different **Activation Functions** on computing physical gradients (force fields) in deep multi-layer perceptrons (MLPs).

Since the HNN must compute higher-order derivatives of the network's output to calculate physical forces, the mathematical properties of the activation function are critical. We conducted extensive ablation studies with the following findings:

1. **The Baseline Champion (Global Sine):** A simple, global `sin(x)` function serves as the optimal baseline (Loss ~ $9.3 \times 10^{-6}$). Its infinite smoothness, global periodicity, and perfectly bounded derivatives make it ideal for maintaining stable gradient flows across the complex chaotic phase space.
2. **The Pitfall of Polynomials (Learnable Polynomial):** We attempted to use a 3rd-order learnable polynomial to explicitly match the HH potential formula. However, this resulted in massive **gradient explosions** (Loss $\to 6 \times 10^8$). The composition of functions in a deep MLP expands the polynomial order exponentially, causing the chain rule to generate numerically unstable gradients.
3. **The Tunnel Vision of Wavelets (Gabor Activation):** We introduced a Gabor wavelet ($e^{-\gamma x^2}\sin(ax)$) to capture localized transient resonances. However, during long-time chaotic predictions, the network learned an excessively large $\gamma$ ($\approx 3.01$), shrinking the receptive field and causing **dead gradients** ("tunnel vision") at the high-energy edges of the phase space.
4. **Frequency Collapse (Dual Adaptive Sin):**
   When explicitly providing two learnable frequencies to capture the non-linear coupling, the network heavily decayed the weight of one frequency, effectively collapsing back to a single frequency optimal for the current energy state.

**Conclusion on Architecture:** In the context of deep MLPs modeling conservative vector fields, trying to enforce structural complexity into the activation layer is sub-optimal. *Simplicity and bounded derivatives (like `sin`) are key.*

---

## 🚀 Next Steps: Physics-Informed Constraints

Having optimized the network architecture, our current focus is addressing the remaining divergence in long-time predictions ($t \to \infty$) near the strong chaotic regime ($E \ge 1/6$).

We are currently implementing the following improvements:
1. **Energy Penalty in Loss Function**: Introducing a strict physical constraint term: $Loss_{energy} = \lambda \cdot MSE(\mathcal{H}_{pred} - E_{initial})$. This will forcefully penalize any energy drift during the network's trajectory rollout.
2. **Symplectic Integrators**: Replacing the standard Euler method data generators with Symplectic Integrators (e.g., Leapfrog) to ensure the training ground truth itself is strictly energy-conserved.

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
```