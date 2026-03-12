# =====================================================
# LSTM STEP 1 : DATA PREPARATION (SYNTHETIC DATA)
# =====================================================

# -----------------------------------------
# STEP 0: Import required libraries
# -----------------------------------------
import pandas as pd
import numpy as np

# -----------------------------------------
# STEP 1: Create synthetic crowd dataset
# -----------------------------------------
data = {
    "hour": [6, 7, 8, 9, 10, 11, 12, 13, 14],
    "crowd_level": [
        "Low",
        "Medium",
        "High",
        "High",
        "Medium",
        "Medium",
        "Low",
        "Low",
        "Medium"
    ]
}

df = pd.DataFrame(data)

print("Original Crowd Data:")
print(df)

# -----------------------------------------
# STEP 2: Convert labels to numeric values
# -----------------------------------------
crowd_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["crowd_encoded"] = df["crowd_level"].map(crowd_mapping)

print("\nEncoded Crowd Data:")
print(df)

# -----------------------------------------
# STEP 3: Create Time-Series Sequences
# -----------------------------------------
def create_sequences(series, window_size):

    X = []
    y = []

    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])

    return np.array(X), np.array(y)


WINDOW_SIZE = 3

crowd_series = df["crowd_encoded"].values

X, y = create_sequences(crowd_series, WINDOW_SIZE)

print("\nInput Sequences (X):")
print(X)

print("\nTarget Values (y):")
print(y)

# -----------------------------------------
# STEP 4: Reshape for LSTM Input
# -----------------------------------------
# Required shape:
# (samples, time_steps, features)

X = X.reshape((X.shape[0], X.shape[1], 1))

print("\nReshaped Input:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------------------
# STEP 5: Save Prepared Data
# -----------------------------------------
np.save("X_train.npy", X)
np.save("y_train.npy", y)

print("\n✅ Data Preparation Completed")
print("Saved files:")
print(" - X_train.npy")
print(" - y_train.npy")