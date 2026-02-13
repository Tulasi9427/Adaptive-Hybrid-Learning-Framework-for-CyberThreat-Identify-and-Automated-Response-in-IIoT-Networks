📌 Project Overview

This project implements a hybrid deep learning intrusion detection and prevention system for Industrial Internet of Things (IIoT) environments using the NSL-KDD dataset.

The framework integrates:
	•	✅ CNN for spatial feature extraction
	•	✅ BiLSTM for temporal dependency learning
	•	✅ Attention Mechanism for feature importance weighting
	•	✅ Reinforcement Learning (Q-Learning) for adaptive intrusion prevention
	•	✅ Hash-Chain Audit Logging for tamper-proof security auditing

The system not only detects cyber-attacks but also dynamically decides mitigation actions such as blocking or isolating malicious sources.

🏗️ System Architecture

🔎 Detection Module
Input Data (41 Features)
        ↓
1D CNN Layers
        ↓
BiLSTM Layers
        ↓
Attention Mechanism
        ↓
Fully Connected Layers
        ↓
Attack Classification

🛡️ Prevention Module
Predicted Attack + Confidence
        ↓
RL-based Q-Learning Policy
        ↓
Action Selection:
  • ALLOW
  • MONITOR
  • RATE_LIMIT
  • BLOCK_IP
  • ISOLATE_DEVICE
        ↓
Hash-Chain Audit Logging

📊 Dataset
	•	Dataset: NSL-KDD
	•	Classes:
	•	Normal
	•	DoS
	•	Probe
	•	R2L
	•	U2R
	•	Features: 41 network traffic features
	•	Split: 70% Train / 15% Validation / 15% Test

  Dataset source:

  🚀 Key Features

🔹 Deep Learning Detection
	•	CNN for feature extraction
	•	BiLSTM for sequential modeling
	•	Attention layer for contextual weighting
	•	Weighted Cross-Entropy for class imbalance
	•	Early stopping & LR scheduling

🔹 Reinforcement Learning Prevention
	•	Q-learning with ε-greedy exploration
	•	Confidence-gated decision making
	•	Severity-aware blocking strategy
	•	Adaptive learning for evolving threats

🔹 Blockchain-style Audit Logging
	•	SHA-256 hash chaining
	•	Immutable decision tracking
	•	Genesis block initialization
	•	Tamper-resistant prevention log

⸻

📈 Performance Metrics

Detection Performance
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-Score
	•	Matthews Correlation Coefficient
	•	Cohen’s Kappa
	•	Per-class detection rate

Prevention Metrics
	1.	Detection Accuracy
	2.	Threat Mitigation Time
	3.	Resource Efficiency
	4.	Scalability
	5.	Adaptability to New Threats
