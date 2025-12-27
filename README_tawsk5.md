
# 📘 Project 5 – Advanced Analytics

**Course:** Introduction to Data Science with Python  
**Name:** Gvantsa Tchuradze  
**Student ID:**  
**Date:** 12/19/2025  
**Honor Code:** *I certify that this work is my own.*

---

## 📌 Project Overview

This project applies advanced machine learning techniques to a real-world e-commerce customer analytics problem. The primary objective is to **predict customer churn**, understand why customers leave, and translate technical findings into **actionable business recommendations**.

The project is divided into three equally weighted tasks, covering the full data science workflow:

1. **Data Preparation & Exploratory Data Analysis (EDA)**
2. **Model Implementation & Evaluation**
3. **Business Insights & Recommendations**

---

## 📂 Dataset Description

- **Source:** Generated using `generate_project5_data.py`  
- **Size:** 1,000 customers  
- **Features:** 30+ attributes including:
  - Demographics (age, gender, income, education)
  - Purchase behavior (frequency, spending, recency)
  - Engagement metrics (email opens, site visits)
  - Satisfaction & loyalty indicators (NPS, referrals)
- **Target Variable:**  
  - `Churned` (0 = Active, 1 = Churned)

---

## ✅ Task 1: Data Preparation & Exploratory Analysis (2%)

**What was done:**

- Loaded and inspected the dataset (shape, types, statistics)
- Identified and handled missing values
- Analyzed churn distribution and class balance
- Conducted Exploratory Data Analysis:
  - Distribution plots for numerical features
  - Correlation heatmap
  - Churn vs. feature comparisons
  - Categorical analysis by membership and location
- Extracted key insights and formed hypotheses
- Preprocessed data for modeling:
  - Removed ID and leakage column (`Churn_Risk_Score`)
  - One-hot encoded categorical variables
  - Applied feature scaling
  - Performed stratified train-test split (80/20)

**Key insights:**

- Long inactivity and low engagement strongly correlate with churn
- Basic members churn more than Premium and VIP customers
- Satisfaction and loyalty metrics are important retention drivers

---

## ✅ Task 2: Model Implementation (Classification) (2%)

**Models Implemented:**

- Logistic Regression (baseline)  
- Decision Tree (max_depth = 5)  
- Random Forest (n_estimators = 100)  
- Support Vector Machine (SVM)  

**Class Imbalance Handling:**

- Applied `class_weight='balanced'` to improve recall
- Compared balanced vs. unbalanced models

**Evaluation Metrics:**

- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrices
- ROC Curves (all models compared)

**Best Model:** Random Forest  

- Highest ROC-AUC  
- Strong balance between precision and recall  
- Robust feature importance interpretation  

**Key churn drivers identified:**

- Days since last purchase  
- Purchase frequency  
- Satisfaction score & NPS  
- Engagement metrics  
- Membership type  

---

## ✅ Task 3: Business Insights & Recommendations (2%)

**Executive Summary:**

- Churn is predictable and actionable  
- Over 50% of customers are churned or high-risk  
- Proactive retention can prevent major revenue loss  

**Business Impact:**

- ~520 at-risk customers  
- Average spend ≈ $2,000  
- Potential revenue at risk ≈ $1,040,000  

**Actionable Recommendations:**

**Immediate (30 days):**

- Target customers inactive >90 days  
- Personalized discounts for low-frequency buyers  
- Support outreach for low satisfaction customers  

**Strategic (6 months):**

- Strengthen loyalty programs  
- Integrate churn prediction into CRM  
- Improve post-purchase engagement  

**ROI Estimation:**

- Retaining just 20% of at-risk customers → ≈ $208,000 saved revenue  

---

## 🛠️ Technologies & Libraries Used

- Python 3.x  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 📁 Repository / Submission Structure

Project5_Gvantsa_Churadze/
│
├── customer_data.csv
├── Project5_Gvantsa_Churadze.ipynb (or .py)
├── README.md
└── requirements.txt (optional)


---

## ✔️ Reproducibility & Best Practices

- Fixed random states (`random_state=42`)  
- No data leakage  
- Clean, documented code  
- All results reproducible  
- Visualizations included  
- Business insights clearly justified  

---

## 🏁 Final Notes

This project demonstrates:

- End-to-end machine learning workflow  
- Strong understanding of classification techniques  
- Effective translation of analytics into business value  
- Professional presentation suitable for both academic and industry contexts  
