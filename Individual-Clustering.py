
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Load data
grades = pd.read_csv('grades.csv')
attendances = pd.read_csv('attendances.csv')
users = pd.read_csv('users.csv')
courses = pd.read_csv('courses.csv')

# Input: Course ID
course_id_input = int(input("Enter course ID:  "))

# Debug: Check total rows and unique course_ids in grades
print(f"\nDebug: Total rows in grades.csv: {len(grades)}")
print(f"Debug: Unique course_ids in grades.csv: {sorted(grades['course_id'].unique())}")

# Filter students in the course
students_in_course = grades[grades['course_id'] == course_id_input].copy()

# Debug: Print number of students after filtering
print(f"Debug: Number of students for course ID {course_id_input}: {len(students_in_course)}")
# Verify course_id in filtered data
if not students_in_course.empty:
    print(f"Debug: Course IDs in filtered data: {students_in_course['course_id'].unique()}")

# Check if any students are found for the course
if students_in_course.empty:
    print(f"\nError: No students found for course ID {course_id_input}. Please check the course ID and try again.\n")
    exit()

# Merge with student names
students_in_course = students_in_course.merge(users[['id', 'name']], left_on='student_id', right_on='id', how='left')
print(f"Debug: Number of students after merging with users: {len(students_in_course)}")

# Calculate attendance percentage
attendance_in_course = attendances[attendances['course_id'] == course_id_input]
attendance_summary = attendance_in_course.groupby('student_id')['status'].value_counts().unstack().fillna(0)
attendance_summary['total_classes'] = attendance_summary.sum(axis=1)
attendance_summary['present_count'] = attendance_summary['present'] + (attendance_summary['late'] * 0.5)  # Late as half-present
attendance_summary['attendance_percentage'] = (attendance_summary['present_count'] / attendance_summary['total_classes']) * 100

# Merge attendance percentage
students_in_course = students_in_course.merge(attendance_summary[['attendance_percentage']], left_on='student_id', right_index=True, how='left')
print(f"Debug: Number of students after merging with attendance: {len(students_in_course)}")

# Handle missing attendance data
students_in_course['attendance_percentage'] = students_in_course['attendance_percentage'].fillna(0)

# Handle missing or invalid grades
grade_columns = ['quiz1', 'quiz2', 'midterm', 'assignments', 'project', 'final']
students_in_course[grade_columns] = students_in_course[grade_columns].fillna(0)
students_in_course[grade_columns] = students_in_course[grade_columns].clip(lower=0)  # Ensure non-negative

# Normalize grades
max_scores = {'quiz1': 10, 'quiz2': 10, 'midterm': 30, 'assignments': 30, 'project': 30, 'final': 60}
for col in grade_columns:
    students_in_course[f'{col}_normalized'] = students_in_course[col] / max_scores[col]

# Calculate weighted total score (final: 40%, midterm/project: 20%, assignments: 10%, quizzes: 5% each)
students_in_course['total_score'] = (
    0.05 * students_in_course['quiz1_normalized'] +
    0.05 * students_in_course['quiz2_normalized'] +
    0.20 * students_in_course['midterm_normalized'] +
    0.10 * students_in_course['assignments_normalized'] +
    0.20 * students_in_course['project_normalized'] +
    0.40 * students_in_course['final_normalized']
)

# Normalize attendance_percentage to 0–1
students_in_course['attendance_normalized'] = students_in_course['attendance_percentage'] / 100

# Prepare features for clustering (total_score, final_normalized, attendance_normalized)
features = students_in_course[['total_score', 'final_normalized', 'attendance_normalized']]

# Handle any remaining NaN in features
features = features.fillna(0)

# Scale features using StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
students_in_course['cluster'] = kmeans.fit_predict(features_scaled)

# Calculate cluster statistics
cluster_stats = students_in_course.groupby('cluster').agg({
    'total_score': ['mean', 'min', 'max', 'count'],
    'final': ['mean', 'min', 'max'],
    'attendance_percentage': ['mean']
}).round(4)

# Print cluster statistics
print("\nCluster Statistics:")
print("Cluster | Count | Total Score (Mean, Min, Max) | Final (Mean, Min, Max) | Attendance % (Mean)")
for cluster in cluster_stats.index:
    stats = cluster_stats.loc[cluster]
    print(f"{cluster}      | {int(stats[('total_score', 'count')]):5d} | "
          f"{stats[('total_score', 'mean')]:.4f}, {stats[('total_score', 'min')]:.4f}, {stats[('total_score', 'max')]:.4f} | "
          f"{stats[('final', 'mean')]:.2f}, {stats[('final', 'min')]:.2f}, {stats[('final', 'max')]:.2f} | "
          f"{stats[('attendance_percentage', 'mean')]:.2f}")

# Dynamically assign performance group labels based on mean total score
cluster_means = students_in_course.groupby('cluster')['total_score'].mean().sort_values(ascending=False)
sorted_clusters = cluster_means.index.tolist()
cluster_mapping = {
    sorted_clusters[0]: 'High performers',     # Highest mean score
    sorted_clusters[1]: 'Average performers',  # Middle mean score
    sorted_clusters[2]: 'At risk students'     # Lowest mean score
}

# Assign initial performance groups
students_in_course['Performance Group'] = students_in_course['cluster'].map(cluster_mapping)

# Refine performance groups to ensure higher total_score/final for High and lower for At risk
def refine_performance_group(row):
    total_score = row['total_score']
    final = row['final']
    
    # High performers: simplified threshold
    if total_score > 0.67 and final > 42:
        return 'High performers'
    # At risk students: refined threshold
    elif (final < 30 and total_score < 0.55) or (total_score < 0.53 and final < 40):
        return 'At risk students'
    # Average performers: default
    else:
        return 'Average performers'

students_in_course['Performance Group'] = students_in_course.apply(refine_performance_group, axis=1)

# Print performance group summary
group_summary = students_in_course.groupby('Performance Group').agg({
    'total_score': ['mean', 'min', 'max', 'count'],
    'final': ['mean', 'min', 'max'],
    'attendance_percentage': ['mean']
}).round(4)

print("\nPerformance Group Summary:")
print("Group              | Count | Total Score (Mean, Min, Max) | Final (Mean, Min, Max) | Attendance % (Mean)")
for group in group_summary.index:
    stats = group_summary.loc[group]
    print(f"{group:17} | {int(stats[('total_score', 'count')]):5d} | "
          f"{stats[('total_score', 'mean')]:.4f}, {stats[('total_score', 'min')]:.4f}, {stats[('total_score', 'max')]:.4f} | "
          f"{stats[('final', 'mean')]:.2f}, {stats[('final', 'min')]:.2f}, {stats[('final', 'max')]:.2f} | "
          f"{stats[('attendance_percentage', 'mean')]:.2f}")

# Get course name
course_name = courses.loc[courses['id'] == course_id_input, 'name'].values[0]

# Save final output, overwriting with error handling
final_columns = ['name', 'student_id', 'quiz1', 'quiz2', 'midterm', 'assignments', 'project', 'final', 'attendance_percentage', 'total_score', 'Performance Group']
output_file = 'clustering_results_updated.csv'
try:
    students_in_course[final_columns].sort_values(by='Performance Group').to_csv(output_file, index=False, mode='w')
    print(f"\nClustering Results for Course: {course_name} (ID: {course_id_input}) saved to {output_file}\n")
except PermissionError:
    print(f"\nError: Cannot write to {output_file}. Ensure the file is not open in another program (e.g., Excel) and try again.\n")
    raise
