# 🚀 SpaceX Launch Analysis & Mission Outcome Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Models-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib%20%7C%20Seaborn-Visualization-11557C?style=for-the-badge&logo=plotly&logoColor=white" />
</p>

<p align="center">
  ✨ <b>Exploring, cleaning, visualizing, and predicting SpaceX mission outcomes with Machine Learning</b> ✨
</p>

---

## 📖 Overview

This project dives into a **SpaceX launch dataset (2006 onward)** 🛰️ to uncover patterns in launches, payloads, vehicles, and mission outcomes — then builds and compares **six classification models** to predict whether a mission succeeds or fails.

---

## 🗂️ Project Workflow

```
📥 Load Data  →  🧹 Clean Data  →  📊 Visualize  →  🛠️ Feature Engineer  →  🤖 Train Models  →  🏆 Compare & Select Best
```

---

## 🧹 1. Data Cleaning

| 🔧 Step | Description |
|---|---|
| 🕒 Parsing | `Launch Date` & `Launch Time` converted to proper datetime formats |
| 🕳️ Missing Values | Filled with `'unknown'`, `'No Failure'`, `'No Landing'`, or median values |
| 🔁 Duplicates | Checked and reported |

**Columns with missing data handled:**
> `Payload Type`, `Payload Mass (kg)`, `Payload Orbit`, `Customer Name`, `Customer Type`, `Customer Country`, `Failure Reason`, `Landing Type`, `Landing Outcome`

---

## 📊 2. Exploratory Data Analysis & Visualizations

A rich set of charts 🎨 was generated to explore the data:

| 📈 Chart | 🔍 Insight |
|---|---|
| 🟩 Mission Outcome Distribution | Countplot of success vs failure |
| 📆 Launches Per Year | Trend of launch frequency over time |
| ⚖️ Payload Mass vs Launch Year | Scatter plot colored by mission outcome |
| 🚀 Vehicle Type Distribution | Bar chart of vehicle usage |
| 🥧 Payload Type Distribution | Pie chart breakdown |
| 📦 Payload Mass by Outcome | Boxplot comparison |
| 🌊 KDE Plot | Payload mass density curve |
| 🐝 Swarm Plot | Payload mass spread by outcome |
| 🎻 Violin Plot | Payload mass distribution by outcome |
| 🔥 Correlation Heatmap | Feature relationships after encoding |
| 🌟 Feature Importance | Random Forest feature ranking |
| 🏁 Model Accuracy Comparison | Bar chart of all model accuracies |

All plots are saved as **high-resolution PNGs** (300 DPI) 🖼️

---

## 🛠️ 3. Feature Engineering

New features crafted to boost model performance:

- 📅 `Launch Year`, `Launch Month`, `Launch Day`
- ⏰ `Launch Hour`, `Launch Minute`
- 🏋️ `Heavy Payload` → flag for payloads over **3000 kg**
- 🔢 Label Encoding applied to all categorical columns

---

## 🤖 4. Machine Learning Models

Six classifiers were trained and evaluated on an **80/20 train-test split** with **StandardScaler** applied:

| 🧠 Model | Emoji |
|---|---|
| Logistic Regression | 📈 |
| Decision Tree | 🌳 |
| Random Forest | 🌲🌲🌲 |
| K-Nearest Neighbors (KNN) | 📍 |
| Support Vector Machine (SVM) | 🛡️ |
| Naive Bayes | 🎲 |

**Metrics computed for each model:**
✅ Accuracy &nbsp; ✅ Precision &nbsp; ✅ Recall &nbsp; ✅ F1 Score &nbsp; ✅ Confusion Matrix &nbsp; ✅ Classification Report

---

## 🏆 5. Model Comparison

A comparison table and bar chart 📊 rank all models by accuracy, and the **best-performing model** is automatically selected:

```python
best_model = comparison.loc[comparison['Accuracy'].idxmax()]
```

🥇 The winning model is printed with its name and accuracy score at the end of the run.

---

## 📦 Requirements

```bash
pip install pandas matplotlib seaborn scikit-learn numpy
```

---

## ▶️ How to Run

1. Place `spaceX.2006.csv` inside the `Csv-all-Datasets/` folder 📁
2. Update the image output paths if needed 🖌️
3. Run the script:
```bash
python spacex_analysis.py
```
4. Check the `image/` folder for all generated charts 🎨 and the console for model metrics 📋

---

## 🌈 Tech Stack

`Python` • `Pandas` • `NumPy` • `Matplotlib` • `Seaborn` • `Scikit-learn`

---

<p align="center">
  Made with ❤️ and 🚀 rocket science
</p>
