import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
df = pd.read_csv("C:\\Users\\GVANTSA\\OneDrive\\Desktop\\datasc\\Project3_Gvantsa_tchuradze\\student_performance.csv")

# set seaborn style
sns.set(style="whitegrid", palette="pastel")

# research questions:
# 1. which factors predict GPA?
# 2. how study habits differ across performance levels?
# 3. does part-time work affect GPA?
# 4. differences by major, gender, year?

# basic GPA distribution plot
plt.figure(figsize=(8,5))
sns.histplot(df['Current_GPA'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Current GPA")
plt.xlabel("Current GPA")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("gpa_distribution.png", dpi=300)
plt.show()

# count of students per major
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='Major', order=df['Major'].value_counts().index, palette='Set2')
plt.title("Number of Students per Major")
plt.xticks(rotation=45)
plt.xlabel("Major")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("students_per_major.png", dpi=300)
plt.show()

# gender distribution
plt.figure(figsize=(5,5))
sns.countplot(data=df, x='Gender', palette=['#66c2a5','#fc8d62'])
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# scatterplot for study hours vs GPA
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Study_Hours_Per_Week', y='Current_GPA',
                hue='Academic_Status', alpha=0.7, palette='Set1')
plt.title("Study Hours vs Current GPA")
plt.xlabel("Study Hours per Week")
plt.ylabel("Current GPA")
plt.legend(title="Academic Status")
plt.tight_layout()
plt.savefig("study_hours_vs_gpa.png", dpi=300)
plt.show()

# attendance vs GPA
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Attendance_Rate', y='Current_GPA',
                hue='Year', alpha=0.7, palette='Set2')
plt.title("Attendance Rate vs Current GPA")
plt.xlabel("Attendance Rate (%)")
plt.ylabel("Current GPA")
plt.legend(title="Year")
plt.tight_layout()
plt.savefig("attendance_vs_gpa.png", dpi=300)
plt.show()

# previous GPA vs current GPA
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Previous_GPA', y='Current_GPA',
                hue='Major', alpha=0.7, palette='tab10')
plt.title("Previous GPA vs Current GPA")
plt.xlabel("Previous GPA")
plt.ylabel("Current GPA")
plt.legend(title="Major", bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig("prev_gpa_vs_current_gpa.png", dpi=300)
plt.show()

# sleep hours vs GPA
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Sleep_Hours', y='Current_GPA',
                hue='Has_Scholarship', palette=['#8da0cb','#fc8d62'])
plt.title("Sleep Hours vs Current GPA")
plt.xlabel("Sleep Hours per Night")
plt.ylabel("Current GPA")
plt.legend(title="Has Scholarship")
plt.tight_layout()
plt.savefig("sleep_vs_gpa.png", dpi=300)
plt.show()

# boxplot of GPA across majors
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Major', y='Current_GPA', palette='Set3')
plt.title("Current GPA Across Majors")
plt.xticks(rotation=45)
plt.xlabel("Major")
plt.ylabel("Current GPA")
plt.tight_layout()
plt.savefig("gpa_by_major.png", dpi=300)
plt.show()

# GPA vs part-time work
plt.figure(figsize=(6,5))
sns.boxplot(data=df, x=(df['Part_Time_Work_Hours']>0), y='Current_GPA',
            palette=['#66c2a5','#fc8d62'])
plt.title("GPA vs Part-Time Work")
plt.xlabel("Has Part-Time Job")
plt.ylabel("Current GPA")
plt.xticks([0,1], ["No", "Yes"])
plt.tight_layout()
plt.savefig("gpa_vs_parttime.png", dpi=300)
plt.show()

# multi-panel figure with several visualizations
fig, axs = plt.subplots(2,2, figsize=(14,10))

sns.histplot(df['Current_GPA'], bins=20, kde=True, color='skyblue', ax=axs[0,0])
axs[0,0].set_title("GPA Distribution")
axs[0,0].set_xlabel("GPA")
axs[0,0].set_ylabel("Count")

sns.boxplot(data=df, x='Year', y='Current_GPA', palette='Set2', ax=axs[0,1])
axs[0,1].set_title("GPA by Year")
axs[0,1].set_xlabel("Year")
axs[0,1].set_ylabel("GPA")

sns.scatterplot(data=df, x='Study_Hours_Per_Week', y='Current_GPA',
                hue='Academic_Status', alpha=0.7, ax=axs[1,0])
axs[1,0].set_title("Study Hours vs GPA")
axs[1,0].set_xlabel("Study Hours")
axs[1,0].set_ylabel("GPA")

sns.scatterplot(data=df, x='Attendance_Rate', y='Current_GPA',
                hue='Gender', ax=axs[1,1])
axs[1,1].set_title("Attendance vs GPA")
axs[1,1].set_xlabel("Attendance Rate (%)")
axs[1,1].set_ylabel("GPA")

plt.tight_layout()
plt.savefig("multi_panel_analysis.png", dpi=300)
plt.show()

# simple printed quantitative insights

# GPA grouped by study hours category
gpa_by_study = df.groupby(pd.cut(df['Study_Hours_Per_Week'],
                                 bins=[0,5,10,15,20,25,40]))['Current_GPA'].mean()
print("Average GPA by Study Hours per Week:\n", gpa_by_study)

# GPA grouped by part-time work
gpa_by_parttime = df.groupby(df['Part_Time_Work_Hours']>0)['Current_GPA'].mean()
print("\nAverage GPA by Part-Time Work:\n", gpa_by_parttime)

# notes for report 
# - students who study more hours tend to have higher GPA
# - attendance is strongly linked to GPA
# - working many hours part-time can reduce GPA
# - scholarship students usually have higher performance
# - sleeping 7-9 hours is linked to slightly better GPA
