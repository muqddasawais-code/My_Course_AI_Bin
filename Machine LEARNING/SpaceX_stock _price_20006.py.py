
# SpaceX Stock Price Forecasting using
# Numpy, Pandas, Seaborn, TensorFlow/Keras
# Models: SimpleRNN, LSTM, GRU

#==========Import Libraries=====
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
print("TensorFlow version:", tf.__version__)


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

from tensorflow.keras.callbacks import EarlyStopping


# ==========================================================
# 1. Load Dataset
# ==========================================================

df = pd.read_csv(r"FINAL ASSESSEMENT PART-2\spaceX_stock_pr.csv.csv")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

print("\nDataset Shape:", df.shape) 


# ==========================================================
# 3. Convert datetime column
# ==========================================================


df['Datetime'] = pd.to_datetime(df['Datetime'])

# Sort values
df = df.sort_values('Datetime')

# Set datetime as index
df.set_index('Datetime', inplace=True)

# Fill missing values if present
df.fillna(method='ffill', inplace=True)


# ==========================================================
# 4. Numpy Operations
# ==========================================================

close_values = np.array(df['Close'])

print("\nNumpy Operations")
print("Mean:", np.mean(close_values))
print("Maximum:", np.max(close_values))
print("Minimum:", np.min(close_values))
print("Standard Deviation:", np.std(close_values))


# ==========================================================
# 5. Exploratory Data Analysis (Seaborn)
# ==========================================================

# Closing Price Trend

plt.figure(figsize=(12,5))

sns.lineplot(
    x=df.index,
    y=df['Close']
)

plt.title("SpaceX Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Closing Price")

plt.xticks(rotation=45)
plt.savefig(r"FINAL ASSESSEMENT PART-2\Images\closing_price_trend.png",
             dpi=300,
    bbox_inches="tight")
plt.show()


# Correlation Heatmap

plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.savefig(r"FINAL ASSESSEMENT PART-2\Images\correlation_heatmap.png",
             dpi=300,
    bbox_inches="tight")        
plt.show()


# ==========================================================
# 6. Normalize Data
# ==========================================================

scaler = MinMaxScaler()
df['features'] = df[['Open','High','Low','Close','Volume']]
scaled_data = scaler.fit_transform(df['features'])      



# ==========================================================
# 7. Create Time Series Sequences
# ==========================================================

def create_sequences(data, sequence_length):

    X = []
    y = []

    for i in range(len(data)-sequence_length):

        X.append(
            data[i:i+sequence_length]
        )

        y.append(
            data[i+sequence_length]
        )

    return np.array(X), np.array(y)


sequence_length = 5

X, y = create_sequences(
    scaled_data,
    sequence_length
)


# ==========================================================
# 8. Train-Test Split
# ==========================================================

split = int(len(X)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# ==========================================================
# 9. Build Model Function
# ==========================================================

def build_model(model_type):

    model = Sequential()

    model.add(
        Input(shape=(X_train.shape[1],1)))

    if model_type=="RNN":

        model.add(
            SimpleRNN(
                units=50,
                activation='tanh'
            )
        )


    elif model_type=="LSTM":

        model.add(
            LSTM(units=50, 
                 activation='tanh'
            )
        )


    elif model_type=="GRU":

        model.add(
            GRU(
                units=50,
                activation='tanh'
            )
        )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model


# ==========================================================
# 10. Train Models
# ==========================================================

results=[]

models=["RNN","LSTM","GRU"]

for model_name in models:

    print("\n")
    print("="*40)
    print(model_name)
    print("="*40)

    model=build_model(model_name)

    early_stop=EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history=model.fit(

        X_train,
        y_train,

        epochs=20,
        batch_size=4,

        validation_split=0.2,

        callbacks=[early_stop],

        verbose=1
    )



    # Prediction

    prediction=model.predict(X_test)

    prediction=scaler.inverse_transform(
        prediction
    )

    actual=scaler.inverse_transform(
    y_test.reshape(-1,1)
)



    # Metrics

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

    r2=r2_score(
        actual,
        prediction
    )


    results.append(
        [model_name,mae,rmse,r2]
    )


    print("\nMAE =",mae)
    print("RMSE =",rmse)
    print("R2 =",r2)


    # Plot prediction

    plt.figure(figsize=(12,5))

    plt.plot(
        actual,
        label='Actual'
    )

    plt.plot(
        prediction,
        label='Predicted'
    )

    plt.title(
        model_name+" Prediction"
    )

    plt.legend()

    plt.savefig(r"FINAL ASSESSEMENT PART-2\Images\\" + model_name + "_prediction.png",
                dpi=300,
                bbox_inches="tight")
    plt.show()



# ==========================================================
# 11. Model Comparison
# ==========================================================

result_df=pd.DataFrame(

    results,

    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]

)

print("\nModel Performance")
print(result_df)


# Bar Plot

plt.figure(figsize=(8,5))

sns.barplot(
    x='Model',
    y='RMSE',
    data=result_df
)

plt.title(
    "RMSE Comparison"
)
plt.savefig(r"FINAL ASSESSEMENT PART-2\Images\rmse_comparison.png",
            dpi=300,
            bbox_inches="tight")

plt.show()


print("\nProgram Executed Successfully")
