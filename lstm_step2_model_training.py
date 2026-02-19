# -----------------------------------------
# STEP 0: Import required libraries
# -----------------------------------------
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------------------
# STEP 1: Prepare the same data again
# (same as previous file, but included here
# so this script runs independently)
# -----------------------------------------
data = {
    "hour": [6, 7, 8, 9, 10, 11, 12, 13, 14],
    "crowd_level": [
        "Low", "Medium", "High", "High",
        "Medium", "Medium", "Low", "Low", "Medium"
    ]
}

df = pd.DataFrame(data)

crowd_mapping = {"Low": 0, "Medium": 1, "High": 2}
df["crowd_encoded"] = df["crowd_level"].map(crowd_mapping)

# -----------------------------------------
# STEP 2: Create time-series sequences
# -----------------------------------------
def create_sequences(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

WINDOW_SIZE = 3
series = df["crowd_encoded"].values

X, y = create_sequences(series, WINDOW_SIZE)

# -----------------------------------------
# STEP 3: Reshape X for LSTM
# IMPORTANT STEP
# -----------------------------------------
# LSTM expects input shape:
# (samples, time_steps, features)

X = X.reshape((X.shape[0], X.shape[1], 1))

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# -----------------------------------------
# STEP 4: Build the LSTM model
# -----------------------------------------
model = Sequential()

# LSTM layer
model.add(
    LSTM(
        units=50,
        activation="relu",
        input_shape=(WINDOW_SIZE, 1)
    )
)

# Output layer
model.add(Dense(1))

# -----------------------------------------
# STEP 5: Compile the model
# -----------------------------------------
model.compile(
    optimizer="adam",
    loss="mse"
)

# -----------------------------------------
# STEP 6: Train the model
# -----------------------------------------
model.fit(
    X,
    y,
    epochs=200,
    verbose=1
)

# -----------------------------------------
# STEP 7: Predict future crowd
# -----------------------------------------
# Take last 3 known crowd values
last_sequence = np.array([[series[-3], series[-2], series[-1]]])
last_sequence = last_sequence.reshape((1, WINDOW_SIZE, 1))

prediction = model.predict(last_sequence)

print("\nPredicted future crowd value:", prediction[0][0])

model.save("lstm_crowd_model.h5")
