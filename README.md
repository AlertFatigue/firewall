# 🔐 LLM-Assisted Suricata Alert Triage

This project demonstrates how a Large Language Model (LLM) can be used as a **post-processing layer** to reduce alert fatigue in Intrusion Detection Systems (IDS), specifically **Suricata**.

Instead of replacing detection, the LLM performs **context-aware triage**, helping prioritize alerts and reduce noise.

---

## 📌 Project Idea

Suricata generates many alerts, including:
- True threats
- False positives
- Benign noisy traffic

This project:
1. Sends alert data to an LLM
2. Adds contextual information (VLAN, device, frequency)
3. Classifies alerts into:
   - High-priority threat
   - Likely noise
4. Suggests tuning strategies (e.g., thresholding repetitive alerts)

---

## 🏗️ Project Structure
