# -----------------------------------------
# STEP 0: Import required libraries
# -----------------------------------------
import pandas as pd
import numpy as np

# -----------------------------------------
# STEP 1: Create synthetic crowd data
# (One station, ordered by time)
# -----------------------------------------
data = {
    "hour": [6, 7, 8, 9, 10, 11, 12, 13, 14],
    "crowd_level": [
        "Low", "Medium", "High", "High",
        "Medium", "Medium", "Low", "Low", "Medium"
    ]
}

df = pd.DataFrame(data)

print("Original crowd data:")
print(df)

# -----------------------------------------
# STEP 2: Convert crowd labels to numbers
# -----------------------------------------
crowd_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["crowd_encoded"] = df["crowd_level"].map(crowd_mapping)

print("\nEncoded crowd data:")
print(df)

# -----------------------------------------
# STEP 3: Create time-series sequences
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

print("\nInput sequences (X):")
print(X)

print("\nTarget values (y):")
print(y)
