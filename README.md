# DRL-Based RIS-Aided Wireless Optimization

This repository contains a deep reinforcement learning (DRL) framework for the joint optimization of
base-station beamforming and reconfigurable intelligent surface (RIS) phase configuration in
downlink multi-user MISO (MU-MISO) wireless systems.

The work focuses on **realistic wireless environments**, explicitly modeling:
- Imperfect channel state information (CSI)
- Hardware-impaired, phase-dependent RIS amplitude response
- Gaussian and impulsive (Bernoulli–Gaussian) noise

The framework was developed as part of a **research internship** and forms the basis of an
ongoing conference paper.

---

## 📌 Problem Overview

Reconfigurable Intelligent Surfaces (RIS) enable programmable control of the wireless environment.
However, practical deployment is challenged by:
- Channel estimation errors
- Hardware non-idealities
- Non-Gaussian impulsive noise

This project formulates the joint beamforming and RIS configuration problem as a **continuous-control
Markov Decision Process (MDP)** and solves it using **Soft Actor-Critic (SAC)**.

---

## 🧠 Methodology

- Joint optimization of:
  - BS beamforming matrix
  - RIS phase shift vector
- Continuous action space handled using SAC
- Reward defined as downlink sum-rate
- Scenario-wise benchmarking under:
  - Ideal environment
  - Mismatched CSI
  - Hardware-impaired RIS
  - β-space exploration strategy

---

## 🧪 Experimental Scenarios

The following environments are evaluated:
1. **Ideal State** – Perfect CSI, ideal RIS
2. **Mismatched Environment** – Imperfect CSI, ideal RIS
3. **Golden Standard** – Perfect CSI, hardware-impaired RIS
4. **β-Space Exploration** – Joint handling of hardware impairment and CSI uncertainty

Both **Gaussian** and **impulsive noise** models are analyzed.

---

## 📊 Results

Key observations:
- DRL converges faster and achieves higher sum-rate under Gaussian noise
- Impulsive noise significantly increases learning variance
- β-space exploration improves robustness under mismatch
- Near-optimal performance is achieved despite realistic impairments

Plots and learning curves are provided in the repository.

---

## 🛠️ Repository Structure

```
.
├── main.py                    # Training loop
├── environment.py             # Wireless system environment
├── SAC.py                     # Soft Actor-Critic implementation
├── Beta_Space_Exp_SAC.py      # β-space exploration agent
├── utils.py                   # Helper functions
├── avg_plot.py                # Result aggregation
├── learning_curve.png         # Learning curves
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## 📄 Research Context

This work was carried out during a research internship at **IIT Indore** and is documented in:
- Internship report
- Conference paper draft (under preparation)

---

## ⚠️ Notes

- Training is computationally intensive
- Convergence under impulsive noise remains an open challenge
- Code is intended for research and experimentation, not production deployment

---

## 📜 License
MIT
