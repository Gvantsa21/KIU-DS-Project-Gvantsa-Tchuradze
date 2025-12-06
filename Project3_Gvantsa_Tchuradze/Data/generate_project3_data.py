"""
Project 3: Data Visualization & EDA - Data Generator
Kutaisi International University
Introduction to Data Science with Python

This script generates realistic student academic performance data
for practicing data visualization and exploratory data analysis.

Instructions:
1. Run this script to generate the CSV file
2. Use this file for all tasks in Project 3
3. Do not modify this script or the generated CSV file before starting your work

Generated file:
- student_performance.csv (500 students with comprehensive academic data)
"""

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 70)
print("Project 3: Data Visualization & EDA - Data Generator")
print("Introduction to Data Science with Python")
print("=" * 70)
print()

# ============================================================================
# PART 1: Generate Student Academic Performance Data
# ============================================================================

print("[1/5] Generating student demographic data...")

n_students = 500

# Demographics
student_ids = [f"STU{str(i + 1).zfill(4)}" for i in range(n_students)]
genders = np.random.choice(["Male", "Female"], n_students, p=[0.52, 0.48])
ages = np.random.randint(18, 25, n_students)

# Academic information
majors = [
    "Computer Science",
    "Engineering",
    "Business",
    "Mathematics",
    "Physics",
    "Biology",
    "Economics",
    "Psychology",
]
major_list = np.random.choice(majors, n_students)

years = np.random.choice(
    ["Freshman", "Sophomore", "Junior", "Senior"],
    n_students,
    p=[0.30, 0.28, 0.25, 0.17],
)

# Scholarships (some correlation with performance)
scholarship_base = np.random.choice([True, False], n_students, p=[0.35, 0.65])

print("[2/5] Generating academic performance data...")

# Study hours per week (will correlate with grades)
study_hours = np.random.gamma(3, 3, n_students)  # Gamma distribution
study_hours = np.clip(study_hours, 2, 40)  # Reasonable range

# Attendance rate (%)
attendance_base = np.random.beta(8, 2, n_students) * 100  # Beta distribution
attendance = np.clip(attendance_base, 40, 100)

# Previous semester GPA (on 4.0 scale)
prev_gpa_base = np.random.beta(5, 2, n_students) * 4.0
prev_gpa = np.clip(prev_gpa_base, 1.5, 4.0)

# Current semester course scores (5 courses)
# Scores influenced by study hours, attendance, and previous GPA
courses = ["Mathematics", "Programming", "Statistics", "English", "Science"]


def generate_score(study_h, attend, prev_gpa):
    """Generate course score based on factors with some randomness"""
    base_score = (
        0.3 * (study_h / 40 * 100)  # Study hours contribution
        + 0.3 * attend  # Attendance contribution
        + 0.3 * (prev_gpa / 4.0 * 100)  # Previous GPA contribution
        + 0.1 * np.random.uniform(60, 100)  # Random factor
    )
    # Add some noise
    score = base_score + np.random.normal(0, 8)
    return np.clip(score, 0, 100)


scores_dict = {}
for course in courses:
    course_scores = [
        generate_score(study_hours[i], attendance[i], prev_gpa[i])
        for i in range(n_students)
    ]
    scores_dict[f"{course}_Score"] = np.round(course_scores, 1)

print("[3/5] Generating behavioral and lifestyle data...")

# Extracurricular activities (hours per week)
extracurricular = np.random.exponential(3, n_students)
extracurricular = np.clip(extracurricular, 0, 20)

# Part-time work hours
work_hours = np.random.choice(
    [0, 5, 10, 15, 20], n_students, p=[0.40, 0.15, 0.20, 0.15, 0.10]
)

# Sleep hours (per night)
sleep_base = np.random.normal(7, 1.5, n_students)
sleep_hours = np.clip(sleep_base, 4, 10)

# Distance from campus (km)
distance = np.random.exponential(8, n_students)
distance = np.clip(distance, 0.5, 50)

# Internet quality at home (1-5 scale)
internet_quality = np.random.choice(
    [1, 2, 3, 4, 5], n_students, p=[0.05, 0.10, 0.25, 0.35, 0.25]
)

print("[4/5] Generating additional academic metrics...")

# Midterm vs Final performance
midterm_avg = np.mean([scores_dict[f"{c}_Score"] for c in courses], axis=0)
# Final tends to be slightly different from midterm
final_diff = np.random.normal(2, 8, n_students)
final_avg = np.clip(midterm_avg + final_diff, 0, 100)

# Assignment submission rate (%)
assignment_rate_base = attendance + np.random.normal(0, 15, n_students)
assignment_rate = np.clip(assignment_rate_base, 20, 100)

# Number of late submissions
late_submissions = np.random.poisson(3, n_students)
late_submissions = np.clip(late_submissions, 0, 20)

# Library visits per semester
library_visits = np.random.negative_binomial(5, 0.3, n_students)
library_visits = np.clip(library_visits, 0, 50)

# Update scholarship based on performance
scholarship = scholarship_base.copy()
for i in range(n_students):
    if midterm_avg[i] >= 85 and prev_gpa[i] >= 3.5:
        scholarship[i] = True
    elif midterm_avg[i] < 60:
        scholarship[i] = False

print("[5/5] Calculating final grades and creating dataset...")

# Calculate current GPA (weighted average of courses)
current_scores = np.array([scores_dict[f"{c}_Score"] for c in courses]).T
current_gpa = (current_scores.mean(axis=1) / 100) * 4.0
current_gpa = np.clip(current_gpa, 0, 4.0)

# Pass/Fail status (passing is >= 60)
passed = current_gpa >= 2.4  # 60/100 = 2.4/4.0


# Academic status
def get_status(gpa, year):
    if gpa >= 3.5:
        return "Honor Roll"
    elif gpa >= 2.4:
        return "Good Standing"
    elif gpa >= 2.0:
        return "Probation"
    else:
        return "Academic Warning"


academic_status = [get_status(current_gpa[i], years[i]) for i in range(n_students)]

# ============================================================================
# PART 2: Create DataFrame
# ============================================================================

print()
print("Creating comprehensive dataset...")

data = {
    "Student_ID": student_ids,
    "Gender": genders,
    "Age": ages,
    "Major": major_list,
    "Year": years,
    "Previous_GPA": np.round(prev_gpa, 2),
    "Current_GPA": np.round(current_gpa, 2),
    "Study_Hours_Per_Week": np.round(study_hours, 1),
    "Attendance_Rate": np.round(attendance, 1),
    "Mathematics_Score": scores_dict["Mathematics_Score"],
    "Programming_Score": scores_dict["Programming_Score"],
    "Statistics_Score": scores_dict["Statistics_Score"],
    "English_Score": scores_dict["English_Score"],
    "Science_Score": scores_dict["Science_Score"],
    "Midterm_Average": np.round(midterm_avg, 1),
    "Final_Average": np.round(final_avg, 1),
    "Assignment_Submission_Rate": np.round(assignment_rate, 1),
    "Late_Submissions": late_submissions,
    "Library_Visits": library_visits,
    "Extracurricular_Hours": np.round(extracurricular, 1),
    "Part_Time_Work_Hours": work_hours,
    "Sleep_Hours": np.round(sleep_hours, 1),
    "Distance_From_Campus": np.round(distance, 1),
    "Internet_Quality": internet_quality,
    "Has_Scholarship": scholarship,
    "Passed": passed,
    "Academic_Status": academic_status,
}

df = pd.DataFrame(data)

# ============================================================================
# PART 3: Save to CSV
# ============================================================================

print()
print("=" * 70)
print("Saving CSV file...")
print("=" * 70)

df.to_csv("student_performance.csv", index=False)
print(f"✓ student_performance.csv saved ({len(df)} students)")

# ============================================================================
# PART 4: Print Dataset Summary
# ============================================================================

print()
print("=" * 70)
print("Dataset Summary")
print("=" * 70)
print()
print(f"Total Students: {len(df)}")
print(f"Columns: {len(df.columns)}")
print()

print("DEMOGRAPHIC DISTRIBUTION:")
print(
    f"  • Gender: {(df['Gender'] == 'Male').sum()} Male, {(df['Gender'] == 'Female').sum()} Female"
)
print(f"  • Age Range: {df['Age'].min()}-{df['Age'].max()} years")
print(f"  • Majors: {df['Major'].nunique()} different majors")
print(f"  • Year Distribution:")
for year in ["Freshman", "Sophomore", "Junior", "Senior"]:
    count = (df["Year"] == year).sum()
    print(f"      - {year}: {count}")
print()

print("ACADEMIC PERFORMANCE:")
print(f"  • Average Current GPA: {df['Current_GPA'].mean():.2f}")
print(f"  • GPA Range: {df['Current_GPA'].min():.2f} - {df['Current_GPA'].max():.2f}")
print(
    f"  • Students Passing: {df['Passed'].sum()} ({df['Passed'].sum() / len(df) * 100:.1f}%)"
)
print(f"  • Honor Roll Students: {(df['Academic_Status'] == 'Honor Roll').sum()}")
print(
    f"  • Students on Probation/Warning: {((df['Academic_Status'] == 'Probation') | (df['Academic_Status'] == 'Academic Warning')).sum()}"
)
print()

print("COURSE SCORES (Average):")
for course in courses:
    avg = df[f"{course}_Score"].mean()
    print(f"  • {course}: {avg:.1f}")
print()

print("STUDY HABITS:")
print(f"  • Average Study Hours/Week: {df['Study_Hours_Per_Week'].mean():.1f}")
print(f"  • Average Attendance Rate: {df['Attendance_Rate'].mean():.1f}%")
print(
    f"  • Average Assignment Submission: {df['Assignment_Submission_Rate'].mean():.1f}%"
)
print(f"  • Average Library Visits: {df['Library_Visits'].mean():.1f}")
print()

print("LIFESTYLE FACTORS:")
print(f"  • Average Sleep Hours: {df['Sleep_Hours'].mean():.1f} hours/night")
print(f"  • Students with Part-Time Work: {(df['Part_Time_Work_Hours'] > 0).sum()}")
print(
    f"  • Average Extracurricular Hours: {df['Extracurricular_Hours'].mean():.1f} hours/week"
)
print()

print("FINANCIAL:")
print(
    f"  • Students with Scholarships: {df['Has_Scholarship'].sum()} ({df['Has_Scholarship'].sum() / len(df) * 100:.1f}%)"
)
print()

print("CORRELATIONS TO INVESTIGATE:")
print("  • Study Hours vs GPA")
print("  • Attendance vs Course Scores")
print("  • Sleep vs Performance")
print("  • Part-Time Work vs Academic Success")
print("  • Distance from Campus vs Attendance")
print("  • Previous GPA vs Current Performance")
print()

print("=" * 70)
print("Data generation complete!")
print("You can now proceed with Project 3: Data Visualization & EDA")
print("=" * 70)
print()
print("VISUALIZATION OPPORTUNITIES:")
print("  • Distribution plots for all numerical variables")
print("  • Comparison plots across majors, years, gender")
print("  • Correlation heatmaps")
print("  • Scatter plots for relationships")
print("  • Box plots for group comparisons")
print("  • Time series (midterm vs final)")
print("  • Multi-dimensional analysis")
print("=" * 70)
print()
print("NOTE: Do NOT modify this CSV file before starting your project.")
print("      Load it as-is and create all visualizations in your code.")
print("=" * 70)
