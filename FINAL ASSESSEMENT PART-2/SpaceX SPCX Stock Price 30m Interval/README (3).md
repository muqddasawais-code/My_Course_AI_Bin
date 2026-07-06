#  SpaceX Stock Price Forecasting

### Predicting the Future of Rocket-Fueled Markets with RNN, LSTM & GRU

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 🚀 Overview

This project builds and compares three flavors of recurrent neural networks — **SimpleRNN**, **LSTM**, and **GRU** — to forecast SpaceX stock closing prices from historical time-series data. It's an end-to-end pipeline: raw CSV → cleaned time series → engineered sequences → trained deep learning models → evaluated and visualized results.

Think of it as a small mission control for stock data: ingest, inspect, launch three different "engines" (RNN/LSTM/GRU), and see which one lands the prediction closest to actual price.

---

## ✨ What This Project Does

| Stage | Description |
|---|---|
| 🧹 **Data Cleaning** | Parses datetime index, sorts chronologically, forward-fills missing values |
| 📊 **Exploratory Analysis** | NumPy-based summary stats (mean, max, min, std) on closing prices |
| 🎨 **Visualization** | Line plot of closing price trend + correlation heatmap across OHLCV features |
| ⚖️ **Normalization** | MinMax scaling of Open, High, Low, Close, Volume |
| 🔁 **Sequence Generation** | Sliding-window sequences (length = 3) to frame the problem as supervised learning |
| 🧠 **Model Training** | Three separate Keras models — SimpleRNN, LSTM, GRU — each with Dropout regularization and early stopping |
| 📈 **Evaluation** | MAE, RMSE, and R² computed per model, with actual-vs-predicted plots |
| 🏆 **Comparison** | Bar chart ranking all three architectures by RMSE |

---

## 🛠️ Tech Stack

- **Data Handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Deep Learning:** `TensorFlow` / `Keras` (`SimpleRNN`, `LSTM`, `GRU`, `Dropout`, `Dense`)
- **Preprocessing & Metrics:** `scikit-learn` (`MinMaxScaler`, `mean_absolute_error`, `mean_squared_error`, `r2_score`)

---

## 📁 Project Structure

```
FINAL ASSESSEMENT PART-2/
│
├── spaceX_stock_pr.csv.csv        # Raw historical stock data
├── Images/
│   ├── closing_price.png          # Closing price trend over time
│   ├── heatmap.png                # Feature correlation heatmap
│   ├── RNN_prediction.png         # SimpleRNN: actual vs predicted
│   ├── LSTM_prediction.png        # LSTM: actual vs predicted
│   ├── GRU_prediction.png         # GRU: actual vs predicted
│   └── RMSE_Comparison.png        # Final model comparison chart
└── spacex_forecasting.py          # Main script
```

---

## 🔬 Methodology

1. **Load & Inspect** — the dataset is read, column names stripped of whitespace, and basic diagnostics (`.head()`, `.info()`, `.isnull().sum()`, `.shape`) confirm data integrity.
2. **Index by Time** — `Datetime` is parsed and set as the DataFrame index, then sorted so the series flows correctly through time.
3. **Feature Selection** — `Open`, `High`, `Low`, `Close`, `Volume` are scaled to a 0–1 range using `MinMaxScaler`, which helps recurrent networks converge faster and more stably.
4. **Windowing** — a custom `create_sequences()` function slides a window of 3 time steps across the data, using each window to predict the *next* closing price.
5. **Train/Test Split** — an 80/20 chronological split (no shuffling — this preserves the time-series structure).
6. **Modeling** — each architecture (`SimpleRNN`, `LSTM`, `GRU`) shares the same skeleton: recurrent layer (16 units, `tanh`) → `Dropout(0.2)` → `Dense(1)`, compiled with `adam` and `mse` loss. `EarlyStopping` (patience = 5) prevents overfitting.
7. **Inverse Transform** — predictions are un-scaled back to real dollar values before computing error metrics, so results are interpretable in actual price terms.
8. **Evaluation** — MAE, RMSE, and R² are calculated for every model, then visualized side-by-side.

---

## 📊 Why These Three Models?

| Model | Strength | Trade-off |
|---|---|---|
| **SimpleRNN** | Lightweight, fast baseline | Struggles with longer dependencies (vanishing gradients) |
| **LSTM** | Strong long-term memory via gates | More parameters, slower to train |
| **GRU** | Similar power to LSTM, simpler gating | Sometimes underperforms on very long sequences |

Comparing all three on the same data gives an honest, apples-to-apples answer to *"which architecture actually earns its complexity here?"* — rather than assuming the fanciest model automatically wins.


---

## 📈 Sample Outputs You'll Get

- 📉 **Closing Price Trend** — how SpaceX's price moved over the observed period
- 🔥 **Correlation Heatmap** — relationships between Open/High/Low/Close/Volume
- 🎯 **Prediction Plots** — actual vs. predicted close price, one per model
- 🏁 **RMSE Comparison Bar Chart** — a clear visual verdict on which model performed best

---

## 💡 Notes & Honest Caveats

- The sequence length (3 time steps) and epoch count (20) are intentionally modest — this is built as a learning/assessment project, not a production trading system.
- Stock price prediction is inherently noisy; these models capture short-term patterns in the data provided but **should not be used for real financial decisions**.
- File paths in the script are currently Windows-style and hardcoded (`FINAL ASSESSEMENT PART-2\...`) — update them to match your own directory structure or use `os.path.join()` for portability across operating systems.
