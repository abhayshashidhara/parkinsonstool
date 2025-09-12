A Machine Learning Based Diagnosis Tool for Parkinson’s Disease

# Description
Parkinson’sTool is an open source AI platform for early Parkinson’s screening. It integrates a convolutional neural network that analyzes spiral drawings together with a retrieval augmented language model that interprets patient history, lifestyle, and symptom inputs. The system produces an evidence backed classification alongside a structured and human readable report.

# Motivation
Timely diagnosis is often delayed by limited specialist access, high consultation costs, and low awareness of early symptoms. This project demonstrates how a hybrid AI approach that combines vision and language can improve accessibility, interpretability, and user engagement outside formal clinical settings while remaining useful to clinicians.

# Outcomes
- **Spiral test classifier (ResNet-18)** achieving ~96.25% accuracy on benchmark data.  
- **RAG pipeline (Sentence-BERT + FAISS)** grounding the language model’s outputs in relevant literature and symptom context.  
- **Fusion module** integrating CNN embeddings with text embeddings, providing a holistic decision and an interpretable narrative report.

# System Architecture
The complete workflow is structured as a dual-modality pipeline:

# Data Input
- Spiral drawings uploaded by the user  
- Structured form capturing demographics, lifestyle, and symptoms  

# Computer Vision Branch (CNN)
- ResNet-18 processes spiral drawings  
- Outputs both a binary classification (PD vs. healthy) and an intermediate feature embedding  

# Language Branch (RAG + LLM)
- Patient form is converted into a text query  
- FAISS index retrieves top literature passages  
- BioClinicalBERT/OpenChat generates a structured report: symptom interpretation, reasoning, and suggested next steps  

# Fusion Module
- CNN embedding and text embedding are concatenated  
- Passed through a shallow multi-layer perceptron for joint reasoning  
- Produces a final PD probability score

