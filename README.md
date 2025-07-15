# Intrusion Detection (CIC-IDS2017) - Attemp1

## Problem Description

This project focuses on building a machine learning system for **multi-class network intrusion detection**, based on the **CIC-IDS2017 dataset**. The goal is to develop a reliable and scalable classifier capable of identifying different types of malicious network traffic alongside normal (benign) activity.

### 📦 Dataset: CIC-IDS2017

The **CIC-IDS2017 dataset** was created by the Canadian Institute for Cybersecurity (CIC) and is widely used in academic and industry research on cybersecurity. Unlike many older datasets (e.g., KDD99), CIC-IDS2017 provides **realistic and up-to-date traffic scenarios** that reflect modern network architectures and attack types.

Key characteristics:
- Includes **7 days of traffic** collected using real human behavior profiles
- Contains over **2.8 million labeled flow records**
- Features more than **80 attributes** per connection, including:
  - Flow duration
  - Packet length statistics
  - Inter-arrival times
  - TCP/UDP flag counts
  - Byte and packet rates
  - Header and payload-based features

The dataset includes the following **attack categories**:
- **DoS** (Denial of Service)
- **DDoS** (Distributed DoS)
- **PortScan**
- **Botnet**
- **BruteForce (SSH, FTP)**
- **Web Attacks (XSS, SQL Injection, Command Injection)**
- **Infiltration**
- **Heartbleed**
- And **Benign** (normal) traffic

This project focuses on **multi-class classification**, where the model must distinguish between these attack types and benign traffic.

---

### 🎯 Goal

To train, deploy, and monitor a **multi-class classifier** that can detect the presence and type of a network intrusion, based on extracted features from network flows.

The core ML model used is an **XGBoost Classifier (XGBClassifier)**, which offers robust performance and handles class imbalance effectively — a common issue in intrusion detection datasets.

---

### ⚙️ Technologies and Architecture

The project is designed to be **production-ready** and fully reproducible. It integrates modern MLOps tools and practices:

- **MLflow**: for experiment tracking and model registry
- **Prefect**: to orchestrate the training pipeline and inference workflows
- **FastAPI**: to serve the model as a RESTful API
- **Docker**: to containerize the API for portability and deployment
- **Terraform**: to provision infrastructure (e.g., GCP Storage, Cloud Run)
- **Google Cloud Platform (GCP)**: to host model artifacts, logs, and API services
- **Evidently (coming up)**: to monitor model performance and detect data drift

---

This project simulates a real-world production environment, applying the full ML lifecycle: **data preprocessing → training → evaluation → deployment → monitoring**, using cloud-native and open-source tools. It aims to demonstrate how machine learning can be effectively applied to improve **network security** through automation and intelligent threat classification.
