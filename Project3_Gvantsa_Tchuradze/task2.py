import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# load the dataset
df = pd.read_csv("C:\\Users\\GVANTSA\\OneDrive\\Desktop\\datasc\\Project3_Gvantsa_tchuradze\\student_performance.csv")

# set simple seaborn style
sns.set(style="whitegrid", palette="Set2")

# histogram of GPA split by gender
plt.figure(figsize=(10,6))
sns.histplot(df, x="Current_GPA", kde=True, hue="Gender", multiple="stack", bins=20)
plt.title("Distribution of Current GPA by Gender")
plt.xlabel("Current GPA")
plt.ylabel("Count")
plt.show()

# violin plot of final average grouped by major
plt.figure(figsize=(12,6))
sns.violinplot(data=df, x="Major", y="Final_Average", inner="point", palette="Set2")
plt.xticks(rotation=45)
plt.title("Final Average Scores by Major")
plt.show()

# boxplot + swarmplot for study hours by academic status
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="Academic_Status", y="Study_Hours_Per_Week", palette="Set3", showmeans=True)
sns.swarmplot(data=df, x="Academic_Status", y="Study_Hours_Per_Week", color=".25", alpha=0.6)
plt.title("Study Hours per Week by Academic Status")
plt.show()

# correlation heatmap of numeric columns
numerical_cols = df.select_dtypes(include=np.number).columns
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Variables")
plt.show()

# pairplot of key variables colored by gender
key_vars = ["Current_GPA", "Study_Hours_Per_Week", "Attendance_Rate", "Sleep_Hours", "Previous_GPA"]
sns.pairplot(df[key_vars + ["Gender"]], hue="Gender", diag_kind="kde", palette="Set2")
plt.suptitle("Pair Plot of Key Academic Metrics", y=1.02)
plt.show()

# regression plot of study hours vs GPA
plt.figure(figsize=(10,6))
sns.regplot(x="Study_Hours_Per_Week", y="Current_GPA", data=df,
            scatter_kws={'alpha':0.6}, line_kws={'color':'red'})

# calculate regression stats
slope, intercept, r_value, p_value, std_err = linregress(df["Study_Hours_Per_Week"], df["Current_GPA"])

plt.title(f"Study Hours vs Current GPA (R² = {r_value**2:.2f})")
plt.xlabel("Study Hours per Week")
plt.ylabel("Current GPA")
plt.show()

# countplot of number of students in each major
plt.figure(figsize=(12,6))
order = df['Major'].value_counts().index
sns.countplot(data=df, x="Major", order=order, palette="Set2")

# add count labels
for i, count in enumerate(df['Major'].value_counts().values):
    plt.text(i, count+2, str(count), ha='center')

plt.xticks(rotation=45)
plt.title("Number of Students per Major")
plt.show()

# barplot of GPA by major and year
plt.figure(figsize=(12,6))
sns.barplot(data=df, x="Major", y="Current_GPA", hue="Year", ci=68, palette="Set3")
plt.xticks(rotation=45)
plt.title("Average GPA by Major and Year")
plt.show()

# facet grid comparing attendance vs GPA by gender and scholarship
g = sns.FacetGrid(df, row="Gender", col="Has_Scholarship", height=4, aspect=1.2)
g.map_dataframe(sns.scatterplot, x="Attendance_Rate", y="Current_GPA", alpha=0.6)
g.set_axis_labels("Attendance Rate (%)", "Current GPA")
g.add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Attendance vs GPA by Gender and Scholarship")
plt.show()

# scatter plot with academic status and top students labeled
plt.figure(figsize=(12,7))
sns.scatterplot(data=df, x="Study_Hours_Per_Week", y="Current_GPA",
                hue="Academic_Status", style="Gender", palette="Set2", s=100)

plt.title("Study Hours vs Current GPA with Academic Status")
plt.xlabel("Study Hours per Week")
plt.ylabel("Current GPA")

# pick top 3 students and label them
top_students = df.nlargest(3, "Current_GPA")
for idx, row in top_students.iterrows():
    plt.text(row["Study_Hours_Per_Week"]+0.2, row["Current_GPA"]+0.02,
             row["Student_ID"], fontsize=10, weight="bold")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("study_hours_vs_gpa.png", dpi=300)
plt.show()
