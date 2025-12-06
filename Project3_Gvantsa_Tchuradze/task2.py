# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Load the generated dataset
df = pd.read_csv("C:\\Users\\GVANTSA\\OneDrive\\Desktop\\datasc\\Project3_Gvantsa_tchuradze\\student_performance.csv")

# Set Seaborn style
sns.set(style="whitegrid", palette="Set2")

# ================================================
# Part A: Distribution Analysis (30%)
# ================================================

# 1. Distribution Plot: Current GPA
plt.figure(figsize=(10,6))
sns.histplot(df, x="Current_GPA", kde=True, hue="Gender", multiple="stack", bins=20)
plt.title("Distribution of Current GPA by Gender", fontsize=16)
plt.xlabel("Current GPA")
plt.ylabel("Count")
plt.show()

# 2. Violin Plot: Final Average by Major
plt.figure(figsize=(12,6))
sns.violinplot(data=df, x="Major", y="Final_Average", inner="point", palette="Set2")
plt.xticks(rotation=45)
plt.title("Final Average Scores by Major", fontsize=16)
plt.show()

# 3. Box Plot + Swarm Plot: Study Hours by Academic Status
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="Academic_Status", y="Study_Hours_Per_Week", palette="Set3", showmeans=True)
sns.swarmplot(data=df, x="Academic_Status", y="Study_Hours_Per_Week", color=".25", alpha=0.6)
plt.title("Study Hours per Week by Academic Status", fontsize=16)
plt.show()

# ================================================
# Part B: Relationship Analysis (40%)
# ================================================

# 1. Correlation Heatmap
numerical_cols = df.select_dtypes(include=np.number).columns
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Variables", fontsize=18)
plt.show()

# 2. Pair Plot: Key numerical variables
key_vars = ["Current_GPA", "Study_Hours_Per_Week", "Attendance_Rate", "Sleep_Hours", "Previous_GPA"]
sns.pairplot(df[key_vars + ["Gender"]], hue="Gender", diag_kind="kde", palette="Set2")
plt.suptitle("Pair Plot of Key Academic Metrics", fontsize=16, y=1.02)
plt.show()

# 3. Regression Plot: Study Hours vs Current GPA
plt.figure(figsize=(10,6))
sns.regplot(x="Study_Hours_Per_Week", y="Current_GPA", data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
slope, intercept, r_value, p_value, std_err = linregress(df["Study_Hours_Per_Week"], df["Current_GPA"])
plt.title(f"Study Hours vs Current GPA (R² = {r_value**2:.2f})", fontsize=16)
plt.xlabel("Study Hours per Week")
plt.ylabel("Current GPA")
plt.show()

# ================================================
# Part C: Categorical Analysis (30%)
# ================================================

# 1. Count Plot: Students per Major
plt.figure(figsize=(12,6))
order = df['Major'].value_counts().index
sns.countplot(data=df, x="Major", order=order, palette="Set2")
for i, count in enumerate(df['Major'].value_counts().values):
    plt.text(i, count+2, str(count), ha='center')
plt.xticks(rotation=45)
plt.title("Number of Students per Major", fontsize=16)
plt.show()

# 2. Grouped Analysis: Average GPA by Major and Year
plt.figure(figsize=(12,6))
sns.barplot(data=df, x="Major", y="Current_GPA", hue="Year", ci=68, palette="Set3")
plt.xticks(rotation=45)
plt.title("Average GPA by Major and Year", fontsize=16)
plt.show()

# 3. Facet Grid: Attendance vs GPA by Gender and Scholarship
g = sns.FacetGrid(df, row="Gender", col="Has_Scholarship", height=4, aspect=1.2)
g.map_dataframe(sns.scatterplot, x="Attendance_Rate", y="Current_GPA", alpha=0.6)
g.set_axis_labels("Attendance Rate (%)", "Current GPA")
g.add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Attendance vs GPA by Gender and Scholarship")
plt.show()

# ================================================
# Advanced Custom Matplotlib Visualization
# ================================================

plt.figure(figsize=(12,7))
sns.scatterplot(data=df, x="Study_Hours_Per_Week", y="Current_GPA",
                hue="Academic_Status", style="Gender", palette="Set2", s=100)
plt.title("Study Hours vs Current GPA with Academic Status", fontsize=18)
plt.xlabel("Study Hours per Week", fontsize=14)
plt.ylabel("Current GPA", fontsize=14)

# Annotate top 3 students with highest GPA
top_students = df.nlargest(3, "Current_GPA")
for idx, row in top_students.iterrows():
    plt.text(row["Study_Hours_Per_Week"]+0.2, row["Current_GPA"]+0.02, row["Student_ID"], fontsize=10, weight="bold")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("study_hours_vs_gpa.png", dpi=300)
plt.show()
