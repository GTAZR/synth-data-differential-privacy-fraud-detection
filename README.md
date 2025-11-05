# Synthetic Data Generation & Differential Privacy for Credit Card Fraud Detection

**Course:** SEP 6DA3 – Data Analytics and Big Data  
**Institution:** McMaster University  
**Semester:** Fall 2025  
**Team:** Group 6  
**Contributors:** Andi Dong, Zhiyu Hu, Foram Brahmbhatt, Jiarui Yang, Linghe Shen, Shannon Chen  
**Instructor:** Prof. Pedro Tondo  

---

## 🎯 Project Overview
This project explores how **synthetic data generation** and **differential privacy (DP)** can improve both *fairness* and *confidentiality* in financial machine-learning workflows.  
Using the **Kaggle Credit Card Fraud Detection** dataset, we balanced the highly imbalanced class distribution through **random oversampling** and applied **Laplace noise injection** to protect sensitive attributes.  
We then trained and compared **Random Forest classifiers** on the original and privacy-enhanced data to quantify the **privacy–utility trade-off**.

---

## 🧠 Objectives
- Understand and handle **class imbalance** through random oversampling.  
- Apply **Laplace differential privacy** to sensitive numerical features.  
- Train and evaluate **Random Forest models** on both datasets.  
- Visualize confusion matrices and compare model metrics.  
- Discuss the ethical, technical, and regulatory implications of privacy-preserving ML.

Dataset source: [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## ⚙️ Workflow Summary

### 1️⃣ Data Exploration (Orange)
- Loaded `creditcard.csv` in **Orange Data Mining**.  
- Visualized the extreme class imbalance:  
  - 284,315 legitimate (99.83 %)  
  - 492 fraudulent (0.17 %)  
- Identified the `Amount` feature as privacy-sensitive due to its skewed distribution.

### 2️⃣ Balancing with Random Oversampling
- Split dataset (80 % train / 20 % test, stratified).  
- Oversampled the minority class in the training set until both classes were equal (227,451 records each).  
- Test set remained unaltered to reflect real-world fraud rates (0.17 %).

### 3️⃣ Applying Differential Privacy
- Added Laplace noise to `Amount` in training data with ε = 0.5 after clipping and standardization.  
- Formula: `X_train_priv['Amount'] += np.random.laplace(0, 1/ε, size)`  
- Objective: obscure exact transaction values while retaining data utility.

### 4️⃣ Model Training and Evaluation
| Model | Data Type | ε | Accuracy | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) |
|:------|:-----------|:--|:----------|:--------------------|:-----------------|:--------------|
| Random Forest (Baseline) | Balanced, no noise | – | 0.9996 | 0.9506 | 0.7857 | 0.8603 |
| Random Forest (DP) | Balanced + Laplace noise | 0.5 | 0.9996 | 0.9506 | 0.7857 | 0.8603 |

**Result:** Moderate noise (ε = 0.5) did *not* degrade performance — Random Forest proved robust to privacy perturbations.

### 5️⃣ Visualization
- Orange Distribution and Box Plot → before/after balancing.  
- Seaborn heatmaps → confusion matrices for baseline vs DP models.  
- Bar plot → Precision, Recall, F1 comparison (Class 1).

---

## 🧩 Key Insights
- **Synthetic oversampling** solved the extreme imbalance, allowing the model to learn fraud patterns effectively.  
- **Differential privacy (Laplace)** protected monetary data without reducing accuracy for moderate ε.  
- **Random Forest robustness:** ensemble learning absorbs noise through feature redundancy.  
- **Privacy–utility trade-off:** lower ε gives stronger privacy but higher distortion; ε ≈ 0.5 balanced both goals.

---

## 🛠️ Technologies Used
- **Python 3.11**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**  
- **Scikit-learn** (Random Forest, train_test_split, metrics)  
- **Orange Data Mining** (for exploratory visualization)  

---

## 🧮 Ethical & Practical Discussion

### 🔸 Privacy–Utility Tradeoff
Smaller ε means stronger privacy but lower model interpretability and stability.  
In our case, ε = 0.5 achieved protection without accuracy loss.

### 🔸 Feature Sensitivity
`Amount` is most sensitive as it directly reveals transaction value.  
Future work could inject noise proportionally based on feature importance or information gain.

### 🔸 Synthetic Data Ethics
Oversampling may bias data if duplicates dominate.  
Safer alternatives include SMOTE or GAN-based generation to preserve statistical realism.

### 🔸 Real-World Applications
A privacy-compliant financial data-sharing pipeline could:
1. Generate synthetic records for rare events (fraud).  
2. Apply differential privacy to sensitive features before external use.  
3. Audit models for data leakage risk to satisfy **GDPR/HIPAA** requirements.

---

## 🧭 Reflection
This lab demonstrated that privacy preservation and accuracy need not be mutually exclusive.  
By combining synthetic oversampling and differential privacy, we achieved a balanced pipeline that supports both **fairness in data representation** and **confidentiality in model training**.  
It also highlighted how real-world data science must account for ethical constraints beyond technical performance.

---

## 📁 Repository Structure
