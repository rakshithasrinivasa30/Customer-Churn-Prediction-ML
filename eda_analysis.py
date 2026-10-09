import pandas as pd

# Load cleaned dataset
data = pd.read_csv("data/cleaned_churn_data.csv")

print("========== DATASET OVERVIEW ==========")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

print("\n========== DATA TYPES ==========")
print(data.dtypes)

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

print("\n========== DUPLICATES ==========")
print("Duplicate rows:", data.duplicated().sum())


# ----------------------------------------
# 1. CHURN DISTRIBUTION
# ----------------------------------------
print("\n========== CHURN DISTRIBUTION ==========")

churn_count = data["Churn Label"].value_counts()
print(churn_count)

churn_percentage = data["Churn Label"].value_counts(normalize=True) * 100
print("\nChurn percentage:")
print(churn_percentage)


# ----------------------------------------
# 2. CONTRACT VS CHURN
# ----------------------------------------
print("\n========== CONTRACT VS CHURN ==========")

contract_churn = pd.crosstab(
    data["Contract"],
    data["Churn Label"],
    normalize="index"
) * 100

print(contract_churn)


# ----------------------------------------
# 3. INTERNET SERVICE VS CHURN
# ----------------------------------------
print("\n========== INTERNET SERVICE VS CHURN ==========")

internet_churn = pd.crosstab(
    data["Internet Service"],
    data["Churn Label"],
    normalize="index"
) * 100

print(internet_churn)


# ----------------------------------------
# 4. TENURE VS CHURN
# ----------------------------------------
print("\n========== AVERAGE TENURE VS CHURN ==========")

tenure_churn = data.groupby(
    "Churn Label"
)["Tenure Months"].mean()

print(tenure_churn)


# ----------------------------------------
# 5. MONTHLY CHARGES VS CHURN
# ----------------------------------------
print("\n========== AVERAGE MONTHLY CHARGES VS CHURN ==========")

charges_churn = data.groupby(
    "Churn Label"
)["Monthly Charges"].mean()

print(charges_churn)


# ----------------------------------------
# 6. PAYMENT METHOD VS CHURN
# ----------------------------------------
print("\n========== PAYMENT METHOD VS CHURN ==========")

payment_churn = pd.crosstab(
    data["Payment Method"],
    data["Churn Label"],
    normalize="index"
) * 100

print(payment_churn)


# ----------------------------------------
# 7. TECH SUPPORT VS CHURN
# ----------------------------------------
print("\n========== TECH SUPPORT VS CHURN ==========")

support_churn = pd.crosstab(
    data["Tech Support"],
    data["Churn Label"],
    normalize="index"
) * 100

print(support_churn)


print("\n========== EDA COMPLETED ==========")