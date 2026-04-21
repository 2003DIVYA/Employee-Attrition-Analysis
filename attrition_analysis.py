import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/employee_attrition.csv")

# -----------------------------------
# First 5 Rows
# -----------------------------------
print("First 5 Rows:\n")
print(df.head())

# -----------------------------------
# Dataset Shape
# -----------------------------------
print("\nDataset Shape:")
print(df.shape)

# -----------------------------------
# Column Names
# -----------------------------------
print("\nColumn Names:")
print(df.columns)

# -----------------------------------
# Missing Values
# -----------------------------------
print("\nMissing Values:\n")
print(df.isnull().sum())

# -----------------------------------
# Duplicate Rows
# -----------------------------------
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# -----------------------------------
# Data Types
# -----------------------------------
print("\nData Types:\n")
print(df.dtypes)

# -----------------------------------
# Summary Statistics
# -----------------------------------
print("\nSummary Statistics:\n")
print(df.describe())

# -----------------------------------
# Graph 1: Attrition Count
# -----------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Attrition", data=df)
plt.title("Employee Attrition Count")
plt.show()

# -----------------------------------
# Graph 2: Attrition by Department
# -----------------------------------
plt.figure(figsize=(8,5))
sns.countplot(x="Department", hue="Attrition", data=df)
plt.title("Attrition by Department")
plt.xticks(rotation=30)
plt.show()

# -----------------------------------
# Graph 3: OverTime vs Attrition
# -----------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="OverTime", hue="Attrition", data=df)
plt.title("OverTime vs Attrition")
plt.show()

# -----------------------------------
# Graph 4: Job Satisfaction vs Attrition
# -----------------------------------
plt.figure(figsize=(8,5))
sns.countplot(x="JobSatisfaction", hue="Attrition", data=df)
plt.title("Job Satisfaction vs Attrition")
plt.show()

# -----------------------------------
# Graph 5: Monthly Income Distribution
# -----------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["MonthlyIncome"], kde=True)
plt.title("Monthly Income Distribution")
plt.show()

from sklearn.preprocessing import LabelEncoder

# -----------------------------------
# Convert Categorical Columns into Numbers
# -----------------------------------

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

print("\nEncoded Dataset Preview:\n")
print(df.head())

from sklearn.model_selection import train_test_split

# -----------------------------------
# Feature Selection
# -----------------------------------

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

print("\nInput Features (X):")
print(X.head())

print("\nTarget Variable (y):")
print(y.head())

# -----------------------------------
# Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

from sklearn.ensemble import RandomForestClassifier

# -----------------------------------
# Build Machine Learning Model
# -----------------------------------

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------
# Model Prediction
# -----------------------------------

predictions = model.predict(X_test)

# -----------------------------------
# Accuracy Score
# -----------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

# -----------------------------------
# Classification Report
# -----------------------------------

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# -----------------------------------
# Confusion Matrix
# -----------------------------------

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, predictions))

# -----------------------------------
# Feature Importance Analysis
# -----------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:\n")
print(importance.head(10))

# -----------------------------------
# Feature Importance Graph
# -----------------------------------

plt.figure(figsize=(10,6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance.head(10)
)

plt.title("Top 10 Important Features for Attrition")
plt.show()