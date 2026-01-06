# DRL-Based RIS-Aided Wireless Optimization

This repository contains a **deep reinforcement learning (DRL) framework** for the joint optimization of
**base-station beamforming** and **reconfigurable intelligent surface (RIS) phase configuration**
in downlink multi-user MISO (MU-MISO) wireless systems.

The framework focuses on **realistic wireless environments**, explicitly modeling:
- Imperfect channel state information (CSI)
- Hardware-impaired, phase-dependent RIS amplitude response
- Gaussian and impulsive (Bernoulli–Gaussian) noise

This work was developed as part of a **research internship at IIT Indore** and forms the basis of an
**ongoing conference paper**.

---

## 📌 Problem Overview

Reconfigurable Intelligent Surfaces (RIS) enable programmable control of the wireless propagation
environment. However, practical deployment is challenged by:
- Channel estimation errors
- Hardware non-idealities
- Non-Gaussian impulsive noise

This project formulates the **joint beamforming and RIS configuration problem** as a
**continuous-control Markov Decision Process (MDP)** and solves it using
**Soft Actor-Critic (SAC)**.

---

## 🧠 Methodology

- Joint optimization of:
  - BS beamforming matrix
  - RIS phase shift vector
- Continuous action space handled using **Soft Actor-Critic (SAC)**
- Reward defined as **downlink sum-rate**
- Scenario-wise benchmarking under:
  - Ideal environment
  - Mismatched CSI
  - Hardware-impaired RIS
  - **β-space exploration strategy**

---

## 🧪 Experimental Scenarios

The following environments are evaluated:

1. **Ideal State** – Perfect CSI, ideal RIS  
2. **Mismatched Environment** – Imperfect CSI, ideal RIS  
3. **Golden Standard** – Perfect CSI, hardware-impaired RIS  
4. **β-Space Exploration** – Joint handling of hardware impairment and CSI uncertainty  

Both **Gaussian** and **impulsive noise** models are analyzed.

---

## 📊 Results & Observations

Key observations from experiments:
- DRL converges faster and achieves higher sum-rate under Gaussian noise
- Impulsive noise significantly increases learning variance
- β-space exploration improves robustness under model mismatch
- Near-optimal performance is achieved despite realistic impairments

Learning curves and result plots are included in the repository.

---

## 🛠️ Repository Structure




## 📂 Repository Structure

```
drl-ris-wireless-optimization/
├── main.py                    # Training loop (SAC-based)
├── environment.py             # Wireless system environment
├── SAC.py                     # Soft Actor-Critic implementation
├── Beta_Space_Exp_SAC.py      # β-space exploration agent
├── utils.py                   # Helper functions
├── avg_plot.py                # Result aggregation
├── learning_curve.png         # Learning curves
├── requirements.txt           # Dependencies
├── baselines/
│  └── sinr-model-training/     # Reproduced DDPG baseline implementation
│     ├── DDPG.py
│     ├── main.py
│     ├── reproduce.py
│     ├── environment.py
│     ├── utils.py
│     ├── requirements.txt
│     └── README.md                # Attribution and usage notes
└── README.md                  # This file
```

---

## 📆 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/garg-khushi/drl-ris-wireless-optimization.git
cd drl-ris-wireless-optimization
```

### 2️⃣ Create and activate a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the SAC-Based Training

To train the proposed SAC-based agent:

```bash
python main.py
```

⚠️ **Training is computationally intensive.**
GPU acceleration is recommended for extended experiments.

---

## 📚 Baseline Implementations

Baseline DRL implementations (DDPG-based) used for reproduction and comparison are provided under:

```bash
baselines/sinr-model-training/
```

These baselines are:

- Clearly isolated from the proposed methods
- Fully attributed to original authors
- Included strictly for research comparison and reproducibility

---

## 📄 Research Context

This work is documented in:

- A detailed internship report
- A conference paper draft (under preparation)

---

## ⚠️ Notes

- Training is computationally expensive
- Convergence under impulsive noise remains an open research challenge
- Code is intended for research and experimentation, not production deployment

---

## 📜 License
MIT License
