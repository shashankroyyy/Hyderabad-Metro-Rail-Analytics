# =====================================================
# LSTM STEP 2 : MODEL TRAINING (SYNTHETIC DATA)
# =====================================================

# -----------------------------------------
# STEP 0: Import required libraries
# -----------------------------------------
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------------------
# STEP 1: Prepare synthetic data again
# (Standalone execution)
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

print("Original Data:")
print(df)

# -----------------------------------------
# STEP 2: Encode crowd labels
# -----------------------------------------
crowd_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["crowd_encoded"] = df["crowd_level"].map(crowd_mapping)

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

series = df["crowd_encoded"].values

X, y = create_sequences(series, WINDOW_SIZE)

# -----------------------------------------
# STEP 4: Reshape for LSTM
# -----------------------------------------
X = X.reshape((X.shape[0], X.shape[1], 1))

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# -----------------------------------------
# STEP 5: Build LSTM Model
# -----------------------------------------
model = Sequential()

model.add(
    LSTM(
        units=50,
        activation="relu",
        input_shape=(WINDOW_SIZE, 1)
    )
)

model.add(Dense(1))

# -----------------------------------------
# STEP 6: Compile Model
# -----------------------------------------
model.compile(
    optimizer="adam",
    loss="mse"
)

# -----------------------------------------
# STEP 7: Train Model
# -----------------------------------------
model.fit(
    X,
    y,
    epochs=200,
    verbose=1
)

# -----------------------------------------
# STEP 8: Predict future crowd
# -----------------------------------------
last_sequence = np.array([
    [series[-3], series[-2], series[-1]]
])

last_sequence = last_sequence.reshape((1, WINDOW_SIZE, 1))

prediction = model.predict(last_sequence)

print("\nPredicted Future Crowd Value:",
      prediction[0][0])

# -----------------------------------------
# STEP 9: Save Model
# -----------------------------------------
model.save("lstm_crowd_model.h5")

print("\n✅ Model saved as lstm_crowd_model.h5")