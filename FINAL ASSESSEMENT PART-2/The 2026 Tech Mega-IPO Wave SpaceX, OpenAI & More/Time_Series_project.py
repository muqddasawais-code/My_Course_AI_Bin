# ==========================================================
# END-TO-END TIME SERIES DEEP LEARNING PROJECT
# Dataset: Tech Mega IPO Dataset
# Libraries: NumPy, Pandas, Seaborn, TensorFlow-Keras
# Models: Simple RNN, LSTM, GRU
# Metrics: MAE, RMSE, R²
# ==========================================================

# ======================
# Import Libraries
# ======================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    SimpleRNN,
    LSTM,
    GRU,
    Dropout,
    Input
)

# ======================
# Set Random Seed
# ======================

np.random.seed(42)
tf.random.set_seed(42)

# ======================
# Load Dataset
# ======================


df = pd.read_csv(r'Csv-all-Datasets\the 2026 tech mega-IPO .csv.csv',delimiter=',')

print("\n========== First 5 Rows ==========")
print(df.head())

print("\n========== Dataset Information ==========")
print(df.info())

print("\n========== Column Names ==========")
print(df.columns.tolist())


# ======================
# Validate Required Columns
# ======================

df.columns = df.columns.str.strip().str.lower()

required_columns = [
    "expected_or_listing_date",
    "ipo_valuation_usd_b"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    print("Missing:", missing)
    exit()

# ======================
# Data Preprocessing
# ======================
# ======================
# Data Preprocessing
# ======================

# Convert date column

df["expected_or_listing_date"] = pd.to_datetime(
    df["expected_or_listing_date"],
    errors='coerce'
)

# Fill missing dates instead of dropping rows

df["expected_or_listing_date"] = (
    df["expected_or_listing_date"]
    .ffill()
    .bfill()
)

# Sort according to time

df = df.sort_values(
    by="expected_or_listing_date"
)

# Fill missing target values

df["ipo_valuation_usd_b"] = (
    df["ipo_valuation_usd_b"]
    .fillna(
        df["ipo_valuation_usd_b"].median()
    )
)

# Select target

target = df["ipo_valuation_usd_b"]

print("\nRows after preprocessing:")
print(len(df))

print("\nMissing values:")
print(df.isnull().sum())





# ======================
# Visualization
# ======================

plt.figure(figsize=(12,6))

sns.lineplot(
    x=df["expected_or_listing_date"],
    y=target
)

plt.title(
    "IPO Valuation Trend"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "IPO Valuation"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()
plt.close()

# ======================
# Data Scaling
# ======================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    target.values.reshape(-1,1)
)

print("\nScaled Shape:",scaled_data.shape)

# ======================
# Create Time Sequences
# ======================

TIME_STEPS = 3


def create_sequences(data,time_steps):

    X=[]
    y=[]

    for i in range(
        len(data)-time_steps
    ):

        X.append(
            data[i:i+time_steps]
        )

        y.append(
            data[i+time_steps]
        )

    return np.array(X),np.array(y)


if len(scaled_data)<=TIME_STEPS:

    raise ValueError(
        "Dataset too small for sequence generation"
    )

X,y=create_sequences(
    scaled_data,
    TIME_STEPS
)

print("\nX Shape:",X.shape)
print("Y Shape:",y.shape)

# ======================
# Train/Test Split
# ======================

train_size=max(
    1,
    int(
        len(X)*0.8
    )
)

X_train=X[:train_size]
X_test=X[train_size:]

y_train=y[:train_size]
y_test=y[train_size:]

print("\nTraining Shape:",X_train.shape)
print("Testing Shape:",X_test.shape)

# ======================
# Model Function
# ======================

def build_model(model_type):

    model=Sequential()

    model.add(

        Input(

            shape=(

                X_train.shape[1],
                X_train.shape[2]
            )
        )
    )

    if model_type=="RNN":

        model.add(

            SimpleRNN(
                32,
                activation='tanh'
            )
        )

    elif model_type=="LSTM":

        model.add(

            LSTM(
                32
            )
        )

    elif model_type=="GRU":

        model.add(

            GRU(
                32
            )
        )

    model.add(
        Dropout(
            0.2
        )
    )

    model.add(
        Dense(
            1
        )
    )

    model.compile(

        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model


# ======================
# Train and Evaluate
# ======================

results=[]

models=[

    "RNN",
    "LSTM",
    "GRU"
]

for model_name in models:

    print("\n")
    print("="*50)

    print(
        "Training:",
        model_name
    )

    print("="*50)

    model=build_model(
        model_name
    )

    history=model.fit(

        X_train,
        y_train,

        epochs=30,
        batch_size=2,

        validation_data=(

            X_test,
            y_test
        ),

        verbose=1
    )

    prediction=model.predict(
        X_test
    )

    prediction=scaler.inverse_transform(
        prediction
    )

    actual=scaler.inverse_transform(
        y_test.reshape(-1,1)
    )

    mae=mean_absolute_error(
        actual,
        prediction
    )

    rmse=np.sqrt(

        mean_squared_error(

            actual,
            prediction
        )
    )

    if len(actual)>1:

        r2=r2_score(

            actual,
            prediction
        )

    else:

        r2=np.nan

    results.append([

        model_name,
        mae,
        rmse,
        r2
    ])

    # ======================
    # Prediction Plot
    # ======================

    plt.figure(
        figsize=(8,5)
    )

    plt.plot(
        actual,
        label="Actual"
    )

    plt.plot(
        prediction,
        label="Predicted"
    )

    plt.title(
        model_name+" Prediction"
    )

    plt.legend()

    plt.show()
    plt.close()

    # ======================
    # Loss Plot
    # ======================

    plt.figure(
        figsize=(8,5)
    )

    plt.plot(
        history.history['loss'],
        label='Training Loss'
    )

    plt.plot(
        history.history['val_loss'],
        label='Validation Loss'
    )

    plt.title(
        model_name+" Loss"
    )

    plt.legend()

    plt.show()
    plt.close()


# ======================
# Results Comparison
# ======================

results_df=pd.DataFrame(

    results,

    columns=[

        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]
)

print("\n")
print("="*50)

print(
    "MODEL COMPARISON"
)

print("="*50)

print(
    results_df
)

# ======================
# Best Model
# ======================

if not results_df.empty:

    best_model=(

        results_df
        .sort_values(
            by="RMSE"
        )
        .iloc[0]
    )

    print("\n")

    print("="*50)

    print(
        "Best Model:",
        best_model["Model"]
    )

    print(
        "RMSE:",
        round(
            best_model["RMSE"],
            3
        )
    )

    print("="*50)

# ======================
# RMSE Comparison Chart
# ======================

plt.figure(
    figsize=(10,5)
)

sns.barplot(

    x="Model",
    y="RMSE",
    data=results_df
)

plt.title(
    "RMSE Comparison"
)

plt.show()
plt.close()