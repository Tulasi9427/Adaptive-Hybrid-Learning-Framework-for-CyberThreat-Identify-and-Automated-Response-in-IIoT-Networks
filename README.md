📌 Project Overview

This project implements a hybrid deep learning intrusion detection and prevention system for Industrial Internet of Things (IIoT) environments using the NSL-KDD dataset.

The framework integrates:<br>
	-	✅ CNN for spatial feature extraction<br>
	-	✅ BiLSTM for temporal dependency learning<br>
	-	✅ Attention Mechanism for feature importance weighting<br>
	-	✅ Reinforcement Learning (Q-Learning) for adaptive intrusion prevention<br>
	-	✅ Hash-Chain Audit Logging for tamper-proof security auditing<br>

The system not only detects cyber-attacks but also dynamically decides mitigation actions such as blocking or isolating malicious sources.

🏗️ System Architecture

🔎 Detection Module
<br>

<div align="center">
Input Data (41 Features)<br>
          ↓<br>
1D CNN Layers<br>
         ↓<br>
BiLSTM Layers<br>
          ↓<br>
Attention Mechanism<br>
         ↓<br>
Fully Connected Layers<br>
          ↓<br>
Attack Classification<br>
</div>

🛡️ Prevention Module
<br>
<div align="center">
Predicted Attack + Confidence<br>
          ↓<br>
RL-based Q-Learning Policy<br>
          ↓<br>
Action Selection:<br>
  • ALLOW<br>
  • MONITOR<br>
  • RATE_LIMIT<br>
  • BLOCK_IP<br>
  • ISOLATE_DEVICE<br>
          ↓<br>
Hash-Chain Audit Logging<br>
</div>

📊 Dataset 
<br>
	•	Dataset: NSL-KDD<br>
	•	Classes:
	•	Normal<br>
	•	DoS<br>
	•	Probe<br>
	•	R2L<br>
	•	U2R<br>
	•	Features: 41 network traffic features<br>
	•	Split: 70% Train / 15% Validation / 15% TestV

  Dataset source:

  🚀 Key Features

🔹 Deep Learning Detection
<br>
	>>CNN for feature extraction<br>
	>>BiLSTM for sequential modeling<br>
	>>Attention layer for contextual weighting<br>
	>>Weighted Cross-Entropy for class imbalance<br>
	>>Early stopping & LR scheduling

🔹 Reinforcement Learning Prevention
  <br>
	•	Q-learning with ε-greedy exploration<br>
	•	Confidence-gated decision making<br>
	•	Severity-aware blocking strategy<br>
	•	Adaptive learning for evolving threats<br>

🔹 Blockchain-style Audit Logging
<br>
	•	SHA-256 hash chaining<br>
	•	Immutable decision tracking<br>
	•	Genesis block initialization<br>
	•	Tamper-resistant prevention log<br>

⸻

📈 Performance Metrics

Detection Performance
<br>
	•	Accuracy<br>
	•	Precision<br>
	•	Recall<br>
	•	F1-Score<br>
	•	Matthews Correlation Coefficient<br>
	•	Per-class detection rate<br>

Prevention Metrics
<br>
	1.	Detection Accuracy<br>
	2.	Threat Mitigation Time<br>
	3.	Resource Efficiency<br>
	4.	Scalability<br>
	5.	Adaptability to New Threats<br>
