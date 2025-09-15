# DA5401 Assignment A4 — GMM-Based Synthetic Sampling for Imbalanced Data

## 📌 Project Overview
This project addresses the *credit card fraud detection problem* using the creditcard.csv dataset (from Kaggle).  
The dataset is *highly imbalanced* — only a tiny fraction of transactions are fraudulent.  

The goal is to build models that can better detect fraudulent transactions by:
1. Establishing a *baseline model* on the imbalanced dataset.
2. Applying a *Gaussian Mixture Model (GMM)* to generate synthetic samples for the minority class.
3. Combining *GMM-based oversampling* with *Clustering-Based Undersampling (CBU)* to create a balanced training set.
4. Comparing the performance of different models on the original imbalanced test set.

---

## 📂 Folder Structure  

├── DA5401_A3_MM22B013.ipynb           # Main Jupyter Notebook  
├── README.md                          # Documentation file  
└── creditcard.csv                     # Dataset

⚠ *Note*: The creditcard.csv dataset (~144 MB) exceeds GitHub’s 100 MB file size limit, so it is not included in this repository.

---

## 📊 Dataset
- *Source*: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- *File*: creditcard.csv  
- *Size*: 284,807 transactions, 492 fraud cases (≈0.172%).  
- *Features*: 30 numerical features (V1..V28, Time, Amount).  
- *Target*: Class (0 = non-fraud, 1 = fraud).  

---
## ⚙ Methodology

### Part A: Baseline Model
- Loaded and analyzed dataset; visualized imbalance.
- Split into train/test sets while maintaining class distribution.
- Trained a *Logistic Regression* classifier on the imbalanced training data.
- Evaluated using *Precision, Recall, F1* (more meaningful than accuracy).

### Part B: GMM-Based Synthetic Sampling
- Explained why *GMM* is more powerful than *SMOTE* for complex, multimodal data.
- Fitted a GMM on the minority (fraud) class.
- Selected the optimal number of components (k) using *BIC/AIC*.
- Generated synthetic fraud samples from the GMM.
- Applied *Clustering-Based Undersampling (CBU)* on the majority class to retain diversity.
- Built two balanced versions:
  1. *GMM-only Oversampling* (minority oversampled to match majority).
  2. *GMM + CBU* (majority reduced + minority augmented).

### Part C: Performance Evaluation
- Trained Logistic Regression on:
  - *Baseline (imbalanced)*  
  - *GMM-only oversampling*  
  - *GMM + CBU balanced*  
- Evaluated on the *original imbalanced test set*.  
- Compared *Precision, Recall, F1, ROC AUC, PR AUC*.  
- Visualized results with bar charts and summary tables.
- Provided recommendation based on empirical results and theory.

---
## 📈 Results (Summary)
- *Baseline*: High accuracy but very low recall (missed many frauds).  
- *GMM-only*: Improved recall significantly, some tradeoff in precision.  
- *GMM + CBU*: Balanced recall and precision with strong F1 and PR AUC.  

👉 *Conclusion*:  
GMM-based synthetic sampling *improves minority detection*.  
Among the approaches, *GMM + CBU* is recommended as it provides better balance between detecting frauds and controlling false alarms.  

---

## 🚀 How to Run
1. Open the Jupyter Notebook (Assignment_A4_GMM.ipynb) in *Google Colab* or Jupyter.  
2. Ensure the dataset creditcard.csv is uploaded or accessible in the working directory.  
3. Run all cells in order:  
   - *Part A* → Baseline model.  
   - *Part B* → GMM implementation, synthetic sampling, and CBU.  
   - *Part C* → Model training, evaluation, comparison, and recommendation.  

---

## by
Adarsh Mahaveer Tare

MM22B016
