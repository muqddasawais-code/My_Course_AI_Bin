# 🚀 Tech Mega IPO Deep Learning Forecasting Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Model](https://img.shields.io/badge/Best_Model-GRU-success)
![Status](https://img.shields.io/badge/Status-Completed-green)

------------------------------------------------------------------------

## 📌 Project Description

This project implements an end-to-end deep learning workflow for
forecasting IPO valuation using the 2026 Tech Mega IPO Dataset.

### Included Workflow

-   Data preprocessing
-   Exploratory Data Analysis (EDA)
-   Feature Engineering
-   Label Encoding
-   MinMax Scaling
-   Time-series sequence generation
-   Deep Learning:
    -   Simple RNN
    -   LSTM
    -   GRU
-   Metrics analysis
-   Best model selection

------------------------------------------------------------------------

## 📊 Final Results

  Metric       Value
  ------------ ---------
  Best Model   GRU
  RMSE         894.514

GRU achieved the lowest prediction error and outperformed the other
models.

------------------------------------------------------------------------

## 🖼 Images

Store generated images in:

Images/ - IPO_Trend.png - Heatmap.png - RNN_prediction.png -
LSTM_prediction.png - GRU_prediction.png - RMSE_Comparison.png

Markdown example:

![IPO Trend](Images/IPO_Trend.png)

![Heatmap](Images/Heatmap.png)

![GRU Prediction](Images/GRU_prediction.png)

![RMSE Comparison](Images/RMSE_Comparison.png)

------------------------------------------------------------------------

## ▶ Run

pip install numpy pandas matplotlib seaborn tensorflow scikit-learn

python ipo_deep_learning.py

------------------------------------------------------------------------

## ✅ Conclusion

GRU produced the best performance with RMSE = 894.514 and successfully
captured temporal patterns in IPO valuation forecasting.
