import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("dataset/Telco-Customer-Churn.csv")

# Display first five rows
print("\nFirst five rows:")
print(df.head())

# Dataset dimensions
print("\nDataset shape:")
print(df.shape)

# Column information
print("\nDataset information:")
df.info()

# Statistical summary
print("\nStatistical summary:")
print(df.describe())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Check churn distribution
print("\nChurn distribution:")
print(df["Churn"].value_counts())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values again
print("\nMissing values after conversion:")
print(df.isnull().sum())

# Count missing TotalCharges
print(
    "\nMissing TotalCharges:",
    df["TotalCharges"].isnull().sum()
)

# Check churn percentages
print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Churn count visualization
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.savefig("eda/churn_distribution.png")
plt.close()

# Churn by contract type
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("eda/churn_by_contract.png")
plt.close()

# Churn by tenure
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure vs Churn")
plt.savefig("eda/tenure_vs_churn.png")
plt.close()

print("\nCode Run. Graphs are saved at /eda")
