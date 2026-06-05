import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor


# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Handle missing values
for df in [train, test]:
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())

# Create time features
for df in [train, test]:

    time_parts = df["timestamp"].str.split(":", expand=True)

    df["hour"] = time_parts[0].astype(int)
    df["minute"] = time_parts[1].astype(int)

# Convert binary columns
for df in [train, test]:

    df["LargeVehicles"] = df["LargeVehicles"].map({
        "Allowed": 1,
        "Not Allowed": 0
    })

    df["Landmarks"] = df["Landmarks"].map({
        "Yes": 1,
        "No": 0
    })

# Features
features = [
    "geohash",
    "day",
    "RoadType",
    "NumberofLanes",
    "LargeVehicles",
    "Landmarks",
    "Temperature",
    "Weather",
    "hour",
    "minute"
]

X = train[features]
y = train["demand"]

X_test = test[features]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Validation Shape:", X_valid.shape)

cat_features = [
    "geohash",
    "RoadType",
    "Weather"
]

model = CatBoostRegressor(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

preds = model.predict(X_valid)

r2 = r2_score(y_valid, preds)

print("\nR2 Score:", r2)
print("Competition Score:", max(0, 100 * r2))

# =========================
# Train on Full Dataset
# =========================

model.fit(
    X,
    y,
    cat_features=cat_features
)

# =========================
# Predict Test Data
# =========================

test_predictions = model.predict(X_test)

# =========================
# Create Submission File
# =========================

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": test_predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nsubmission.csv created successfully!")

