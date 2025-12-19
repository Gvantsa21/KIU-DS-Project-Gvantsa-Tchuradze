# 📊 Project 3 — Data Visualization & Exploratory Data Analysis  
### *Introduction to Data Science with Python*  
**Name:** Gvantsa Tchuradze 
**Student ID:**
**Date:** 12/06/2025 
**Honor Code:** *I certify that this work is my own.*

---

## 📁 Project Description
This project explores student academic performance using **Python**, **Matplotlib**, **Seaborn**, and **Exploratory Data Analysis (EDA)**.  
The dataset includes demographic variables, academic indicators, study habits, and lifestyle factors for 500 students.

The project contains 3 major tasks:

- **Task 1:** Matplotlib fundamentals  
- **Task 2:** Seaborn & statistical visualization  
- **Task 3:** Full data story + insights + advanced EDA dashboard  

All visualizations include proper titles, labels, legends, readable text, colorblind-friendly palettes, and saved images.

---

# 📂 Repository Structure

```plaintext
KIU-DS-Project3-YourName/
│
├── Task1    
│
├── Task2     
│
├── Task3    
│
├── data/
│   └── student_performance.csv    # Dataset (required)
│
├── figures/                       # Optional: saved visualization images
│   ├── gpa_distribution.png
│   ├── study_hours_vs_gpa.png
│   ├── major_gpa_boxplot.png
│   └── ...
│
├── README.md
└── requirements.txt
```

---

# 🧪 Dataset
The dataset is generated from the provided instructor script and contains **27 features**, including:

- **Demographics:** Gender, Age, Major, Academic Year  
- **Academics:** Current GPA, Previous GPA, Final Average  
- **Behavior:** Study Hours, Attendance Rate  
- **Lifestyle:** Sleep Hours, Health Score  
- **Status Indicators:** Scholarship, Part-Time Work  

The CSV file must be included inside the **data/** folder exactly as shown.

---

# 📘 Task Overview

---

## ✅ **Task 1 — Matplotlib Fundamentals**
Included visualizations:

- Line plot (Study Hours vs GPA)
- Scatter plot (Attendance vs Final Average, colored by Major)
- Histogram of GPA with mean/median lines
- Bar chart (Average GPA by Major)
- Boxplot (Subject score comparison)
- 2×2 Subplot figure
- Clean professional visualization with consistent style
- Saved images in **figures/**

---

## 🎨 **Task 2 — Seaborn Statistical Visualizations**

- Distribution & KDE plots  
- Violin plot (Major vs Final Average)  
- Swarm/box plots  
- Correlation heatmap  
- Pairplot with hue  
- Regression plot + R²  
- Countplot (Majors)  
- Multi-category comparison (Major × Year)  
- FacetGrid split by Gender & Scholarship  

---

## 📊 **Task 3 — Full Visual Story**

### Contains three parts:

### **A. Research Questions**
Example included questions:
- What factors have the strongest impact on GPA?
- How do study habits differ across performance groups?
- Do scholarship students perform better?
- Does part-time work reduce academic performance?

---

### **B. Dashboard (6–10 plots)**
Organized into:

#### 🔹 Performance Overview
- GPA distribution  
- Demographic composition  
- Average scores per subject  

#### 🔹 Academic Predictors
- Study hours vs GPA  
- Attendance vs Final Average  
- Sleep hours vs GPA  
- Previous GPA vs Current GPA  

#### 🔹 Group Comparisons
- GPA by Major  
- GPA by Academic Year  
- Scholarship status difference  
- Impact of part-time work  

---

### **C. Insights & Recommendations (~300 words)**  
Interpretation includes:
- Key correlations  
- Strongest predictors  
- Comparison insights  
- Actionable recommendations  
- Limitations & next steps  

---

# ▶️ Running Instructions

## **Option 1 — Jupyter Notebook**
```
jupyter notebook Project3_YourName.ipynb
```

## **Option 2 — Python Script**
```
python Project3_YourName.py
```

Both methods will display all plots automatically.

---


# 📈 Key Findings (Summary)

- Students who study more than **15 hours/week** have significantly higher GPAs  
- Attendance rate shows a **strong positive correlation** with GPA  
- Sleeping **7–9 hours** leads to better academic outcomes  
- Previous GPA is the **best predictor** of current GPA  
- Scholarship students maintain higher academic performance  
- Part-time work slightly reduces GPA due to lower study time  
- Engineering majors show high workload and wider GPA variance  

---

# 🧩 Limitations
- Synthetic dataset may not perfectly model real-world data  
- Some relationships may be non-linear  
- External factors not included (motivation, stress, teaching quality)  

---

# 🎯 Conclusion
This project demonstrates how visualization and exploratory data analysis reveal valuable insights about student performance. Using Matplotlib, Seaborn, and statistical graphics, we identify key factors that shape academic outcomes and propose evidence-based recommendations.

---

