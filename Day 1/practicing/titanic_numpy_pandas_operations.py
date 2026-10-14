from pathlib import Path
import csv
import numpy as np
import pandas as pd
DATA_FILE = Path(__file__).with_name("train_titanic.csv")
OUTPUT_DIR = Path(__file__).with_name("titanic_analysis_output")
def numpy_analysis():
    dtype = [
        ("PassengerId", "i4"), ("Survived", "i1"), ("Pclass", "i1"),
        ("Sex", "U6"), ("Age", "f8"), ("SibSp", "i1"), ("Parch", "i1"),
        ("Fare", "f8"), ("Embarked", "U1"),
    ]
    with DATA_FILE.open(newline="", encoding="utf-8") as source:
        rows = csv.DictReader(source)
        records = [
            (int(r["PassengerId"]), int(r["Survived"]), int(r["Pclass"]), r["Sex"],
             float(r["Age"]) if r["Age"] else np.nan, int(r["SibSp"]), int(r["Parch"]),
             float(r["Fare"]), r["Embarked"])
            for r in rows
        ]
    data = np.array(records, dtype=dtype)
    age, fare = data["Age"], data["Fare"]
    family_size = data["SibSp"] + data["Parch"] + 1
    age_filled = np.where(np.isnan(age), np.nanmedian(age), age)
    features = np.column_stack((data["Pclass"], age_filled, fare, family_size, data["Survived"]))
    print("\n========== NUMPY OPERATIONS ==========")
    print("Columns:", data.dtype.names)
    print("First 3 rows:\n", data[:3])
    print("Missing ages:", np.isnan(age).sum())
    print("Age mean / median:", round(np.nanmean(age), 2), "/", np.nanmedian(age))
    print("Fare min / max / mean:", np.nanmin(fare), np.nanmax(fare), round(np.nanmean(fare), 2))
    female = data["Sex"] == "female"
    children = age < 18
    print("Female survival rate:", round(data["Survived"][female].mean(), 3))
    print("Child survival rate:", round(data["Survived"][children].mean(), 3))
    labels, counts = np.unique(data["Embarked"], return_counts=True)
    print("Embarked counts:", dict(zip(labels.tolist(), counts.tolist())))
    for passenger_class in np.unique(data["Pclass"]):
        group = data["Pclass"] == passenger_class
        print(f"Class {passenger_class}: count={group.sum()}, survival={data['Survived'][group].mean():.3f}")
    top_fare_indices = np.argsort(fare)[-5:][::-1]
    print("Top five fares (PassengerId, Fare):\n", np.column_stack((data["PassengerId"][top_fare_indices], fare[top_fare_indices])))
    print("Passengers travelling alone:", np.sum(family_size == 1))
    print("Fare z-scores, first 5:", np.round((fare[:5] - np.nanmean(fare)) / np.nanstd(fare), 3))
    print("Age percentiles (25, 50, 75):", np.round(np.nanpercentile(age, [25, 50, 75]), 2))
    print("Fare percentiles (25, 50, 75, 95):", np.round(np.percentile(fare, [25, 50, 75, 95]), 2))
    print("Fare histogram (5 bins):", np.histogram(fare, bins=5)[0].tolist())
    print("Fare values above 95th percentile:", int(np.sum(fare > np.percentile(fare, 95))))
    print("Correlation matrix [Pclass, Age, Fare, FamilySize, Survived]:\n", np.round(np.corrcoef(features, rowvar=False), 3))
def pandas_analysis():
    df = pd.read_csv(DATA_FILE)
    cleaned = df.copy()
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    })
    duplicate_rows = int(df.duplicated().sum())
    cleaned["Embarked"] = cleaned["Embarked"].fillna(cleaned["Embarked"].mode().iat[0])
    cleaned["AgeFilled"] = cleaned.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda values: values.fillna(values.median())
    )
    cleaned["AgeFilled"] = cleaned["AgeFilled"].fillna(cleaned["Age"].median())
    cleaned["FamilySize"] = cleaned["SibSp"] + cleaned["Parch"] + 1
    cleaned["IsAlone"] = cleaned["FamilySize"].eq(1)
    cleaned["FareZScore"] = (cleaned["Fare"] - cleaned["Fare"].mean()) / cleaned["Fare"].std(ddof=0)
    cleaned["AgeGroup"] = pd.cut(cleaned["AgeFilled"], [-1, 17, 59, float("inf")], labels=["child", "adult", "senior"])
    cleaned["FareBand"] = pd.qcut(cleaned["Fare"], q=4, duplicates="drop")
    cleaned["Title"] = cleaned["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    cleaned["Deck"] = cleaned["Cabin"].str[0].fillna("Unknown")
    cleaned["TicketGroupSize"] = cleaned.groupby("Ticket")["Ticket"].transform("size")
    cleaned["FarePerTicket"] = cleaned["Fare"] / cleaned["TicketGroupSize"]
    cleaned["SexCode"] = cleaned["Sex"].map({"female": 1, "male": 0})
    fare_q1, fare_q3 = cleaned["Fare"].quantile([0.25, 0.75])
    fare_iqr = fare_q3 - fare_q1
    fare_upper_fence = fare_q3 + 1.5 * fare_iqr
    fare_outliers = cleaned.loc[cleaned["Fare"] > fare_upper_fence, ["PassengerId", "Name", "Fare"]]
    class_summary = cleaned.groupby("Pclass", observed=True).agg(
        passengers=("PassengerId", "size"), survival_rate=("Survived", "mean"),
        mean_fare=("Fare", "mean"), mean_age=("Age", "mean"),
    ).round(3)
    survival_by_class_sex = pd.crosstab(
        cleaned["Pclass"], cleaned["Sex"], values=cleaned["Survived"], aggfunc="mean"
    ).round(3)
    survival_by_age_embarked = cleaned.pivot_table(
        index="AgeGroup", columns="Embarked", values="Survived", aggfunc="mean", observed=True
    ).round(3)
    correlations = cleaned[["Pclass", "AgeFilled", "Fare", "FamilySize", "TicketGroupSize", "SexCode", "Survived"]].corr().round(3)
    OUTPUT_DIR.mkdir(exist_ok=True)
    cleaned.to_csv(OUTPUT_DIR / "titanic_cleaned_features.csv", index=False)
    missing.to_csv(OUTPUT_DIR / "missing_values.csv")
    class_summary.to_csv(OUTPUT_DIR / "class_summary.csv")
    survival_by_class_sex.to_csv(OUTPUT_DIR / "survival_by_class_and_sex.csv")
    survival_by_age_embarked.to_csv(OUTPUT_DIR / "survival_by_age_group_and_embarked.csv")
    correlations.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    fare_outliers.to_csv(OUTPUT_DIR / "fare_outliers.csv", index=False)
    print("\n========== PANDAS OPERATIONS ==========")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("First 3 rows:\n", df.head(3))
    print("Data types:\n", df.dtypes)
    print("Duplicate rows:", duplicate_rows)
    print("Missing values:\n", missing)
    print("Numerical summary:\n", df[["Age", "Fare", "SibSp", "Parch"]].describe())
    print("Female survival rate:", round(df.loc[df["Sex"].eq("female"), "Survived"].mean(), 3))
    print("Children (<18):", len(df[df["Age"] < 18]))
    print("Embarked counts:\n", df["Embarked"].value_counts(dropna=False))
    print("Class summary:\n", class_summary)
    print("Survival rate by class and sex:\n", survival_by_class_sex)
    print("Survival rate by age group and embarkation:\n", survival_by_age_embarked)
    print("Top five fares:\n", df.nlargest(5, "Fare")[["PassengerId", "Name", "Fare"]])
    print("Age-group counts:\n", cleaned["AgeGroup"].value_counts())
    print("Fare outliers above IQR upper fence (%.2f): %d" % (fare_upper_fence, len(fare_outliers)))
    print("Correlation matrix:\n", correlations)
    print("Exported analysis files to:", OUTPUT_DIR)
if __name__ == "__main__":
    numpy_analysis()
    pandas_analysis()