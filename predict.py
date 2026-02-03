import numpy as np
import pandas as pd
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from extract import fetch_passenger_counts_df


TIME_STEP_SECONDS = 5
PREDICTION_MINUTES = 10
PREDICTION_STEPS = (PREDICTION_MINUTES * 60) // TIME_STEP_SECONDS

WINDOW_SIZE = 60        # past 5 minutes (60 × 5s)
TRAIN_SPLIT = 0.8
EPOCHS = 25
BATCH_SIZE = 32

# LOAD DATA
df = fetch_passenger_counts_df()

if len(df) < WINDOW_SIZE + 10:
    raise ValueError("Not enough data to train the GRU model.")

start_time = df["timestamp"].iloc[0]
end_time = df["timestamp"].iloc[-1]
total_seconds = (end_time - start_time).total_seconds()
total_minutes = total_seconds / 60

print("Recording duration:")
print(f"  Start: {start_time}")
print(f"  End:   {end_time}")
print(f"  Total: {total_seconds:.2f} seconds ({total_minutes:.2f} minutes)\n")


values = df["count"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_values, WINDOW_SIZE)


# TRAIN / PREDICTION SPLIT

split_idx = int(len(X) * TRAIN_SPLIT)
X_train, y_train = X[:split_idx], y[:split_idx]
X_predict_seed = X[-1:]   

print(f"Training samples: {len(X_train)}")
print(f"Prediction seed window size: {X_predict_seed.shape}\n")


# BUILD GRU MODEL

model = Sequential([
    Input(shape=(WINDOW_SIZE, 1)),
    GRU(64, return_sequences=True),
    GRU(32),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

# TRAIN MODEL
model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# FUTURE PREDICTION (10 MIN)

future_scaled = []
current_window = X_predict_seed.copy()

for _ in range(PREDICTION_STEPS):
    next_pred = model.predict(current_window, verbose=0)
    future_scaled.append(next_pred[0, 0])

    # slide window
    current_window = np.roll(current_window, -1, axis=1)
    current_window[0, -1, 0] = next_pred[0 , 0]

# inverse scaling
future_values = scaler.inverse_transform(
    np.array(future_scaled).reshape(-1, 1)
).flatten()


# BUILD FUTURE TIMESTAMPS

last_timestamp = df["timestamp"].iloc[-1]

future_timestamps = [
    last_timestamp + timedelta(seconds=TIME_STEP_SECONDS * (i + 1))
    for i in range(PREDICTION_STEPS)
]

prediction_df = pd.DataFrame({
    "timestamp": future_timestamps,
    "predicted_count": future_values.round(2)
})

# =========================
# OUTPUT
# =========================
# print("\nNext 10 minutes passenger count prediction (5s interval):")
# print(prediction_df.head(10))

# print("\nLast prediction:")
# print(prediction_df.tail(1))

OUTPUT_FILE = "predictions_10min.csv"

prediction_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nFull 10-minute prediction saved to: {OUTPUT_FILE}")
print(f"Total prediction rows: {len(prediction_df)}")
