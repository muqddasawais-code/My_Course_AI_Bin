import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import ( LabelEncoder,MinMaxScaler)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    SimpleRNN,
    LSTM,
    GRU,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping


#===========Random Seed=======

np.random.seed(42)
tf.random.set_seed(42)


# =======Load Dataset=====
df = pd.read_csv(r"Csv-all-Datasets\the 2026 tech mega-IPO .csv.csv",delimiter=',')
print("\nDataset Shape")
print(df.shape)
print("\nColumns")
print(df.columns.tolist())
print("\nMissing Values")
print(df.isnull().sum())

# ===========Data Cleaning==============

df.columns=df.columns.str.strip()
df["expected_or_listing_date"]=pd.to_datetime(df["expected_or_listing_date"],errors='coerce')

df["expected_or_listing_date"]=(
    df["expected_or_listing_date"]
    .ffill()
    .bfill()
)

df=df.sort_values("expected_or_listing_date")

# ========Feature Engineering==========

current_year=2026

if "year_founded" in df.columns:

    df["company_age"]=(
        current_year-
        df["year_founded"]
    )

else:

    df["company_age"]=df["years_to_ipo"]


# ========Funding efficiency===========

df["funding_efficiency"]=(
    df["ipo_valuation_usd_b"]/
    (
        df["amount_raised_usd_b"]+0.0001
    )
)

# =======Revenue efficiency==========

df["revenue_efficiency"]=(
    df["ipo_valuation_usd_b"]/
    (
        df[
        "latest_annual_revenue_or_arr_usd_b"
        ]+0.0001
    )
)

#======Valuation growth======

df["valuation_growth"]=(
    df["ipo_valuation_usd_b"]
    -
    df["last_private_valuation_usd_b"]
)

# ======Funding ratio==========

df["funding_ratio"]=(
    df["amount_raised_usd_b"]
    /
    (
        df["ipo_valuation_usd_b"]
        +0.0001
    )
)

df.fillna(0,inplace=True)



# =============Label Encoding================

encoder=LabelEncoder()

categorical_columns=[

"company",
"ticker",
"sector",
"is_ai_company",
"ipo_status",
"revenue_type",
"headquarters_country",
"lead_investor_or_backer"

]

for col in categorical_columns:

    if col in df.columns:

        df[col]=encoder.fit_transform(
            df[col].astype(str)
        )


# =============EDA=================

plt.figure(figsize=(12,5))
sns.lineplot(x=df["expected_or_listing_date"],y=df["ipo_valuation_usd_b"])
plt.title("IPO Valuation Trend")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"FINAL ASSESSEMENT PART-2\The 2026 Tech Mega-IPO Wave SpaceX, OpenAI & More\Images.i\ipo_valuation_trend.png",
            dpi=300,bbox_inches='tight')
plt.show()
plt.close()


#==========Heatmap===========

plt.figure(figsize=(16,12))
numeric_df=df.select_dtypes(include=np.number)

sns.heatmap(numeric_df.corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(r"FINAL ASSESSEMENT PART-2\The 2026 Tech Mega-IPO Wave SpaceX, OpenAI & More\Images.i\correlation_heatmap.png",
            dpi=300,bbox_inches='tight')
plt.show()
plt.close()


# ============Feature Selection=================

features=df[[

"company",
"sector",
"is_ai_company",
"ipo_status",
"last_private_valuation_usd_b",
"amount_raised_usd_b",
"latest_annual_revenue_or_arr_usd_b",
"valuation_to_revenue_multiple",
"years_to_ipo",
"company_age",
"funding_efficiency",
"revenue_efficiency",
"valuation_growth",
"funding_ratio"

]]

target=df[
    "ipo_valuation_usd_b"
]


# ==========Scaling============

feature_scaler=MinMaxScaler()

target_scaler=MinMaxScaler()

X_scaled=feature_scaler.fit_transform(
    features
)

y_scaled=target_scaler.fit_transform(

    target.values.reshape(-1,1)

)

print("\nSamples after preprocessing:", len(df))

# ========Sequence Creation================

def create_sequences(
    X,
    y,
    sequence_length
):

    X_seq=[]
    y_seq=[]

    for i in range(
        len(X)-sequence_length
    ):

        X_seq.append(
            X[
                i:i+sequence_length
            ]
        )

        y_seq.append(
            y[
                i+sequence_length
            ]
        )

    return (
        np.array(X_seq),
        np.array(y_seq)
    )


sequence_length=min(
    2,
    len(X_scaled)-1
)

print(
    "\nSequence Length:",
    sequence_length
)

X,y=create_sequences(X_scaled,y_scaled,sequence_length)
print("\nX Shape:",X.shape)
print("Y Shape:",y.shape)


# =======Train Test Split==============

split=max(
    1,
    int(
        len(X)*0.80
    )
)

X_train=X[:split]
X_test=X[split:]

y_train=y[:split]
y_test=y[split:]


# ==========Build Model==============

def build_model(model_type):

    model = Sequential()

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
                units=16
            )
        )

    elif model_type=="LSTM":

        model.add(
            LSTM(
                units=16
            )
        )

    elif model_type=="GRU":

        model.add(
            GRU(
                units=16
            )
        )

    model.add(
        Dropout(0.1)
    )

    model.add(
        Dense(8,
        activation='relu')
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
# ==========Training + Evaluation==============

results=[]

models=["RNN","LSTM","GRU"]

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

    stop=EarlyStopping(

        monitor='val_loss',

        patience=5,

        restore_best_weights=True
    )

    history=model.fit(X_train,y_train,epochs=30,batch_size=2,validation_split=0.1,callbacks=[stop],verbose=1)
    prediction=model.predict(X_test)
    prediction=target_scaler.inverse_transform(prediction)
    actual=target_scaler.inverse_transform(y_test)

    mae=mean_absolute_error(actual,prediction)
    rmse=np.sqrt(

        mean_squared_error(actual,prediction))

    if len(actual)>1:

        r2=r2_score(actual,prediction)

    else:

        r2=np.nan
    results.append([model_name,mae,rmse,r2])

    plt.figure(figsize=(8,5))
    plt.plot(actual.flatten(),label="Actual")
    plt.plot(prediction.flatten(),label="Prediction")
    plt.title(model_name+" Prediction")
    plt.legend()
plt.savefig(r"FINAL ASSESSEMENT PART-2\The 2026 Tech Mega-IPO Wave SpaceX, OpenAI & More\Images.i\'+model_name+'_Prediction.png",
            dpi=300,bbox_inches='tight')
plt.show()
plt.close()


#============Result Analysis==============

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
print("MODEL PERFORMANCE")
print("="*50)
print(results_df)

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


# =======RMSE Graph=======

plt.figure(figsize=(8,5))
sns.barplot(x="Model",y="RMSE",data=results_df)
plt.title("RMSE Comparison")
plt.savefig(r"FINAL ASSESSEMENT PART-2\The 2026 Tech Mega-IPO Wave SpaceX, OpenAI & More\Images.i\RMSE_Comparison.png",
            dpi=300,bbox_inches='tight')
plt.show()
plt.close()

print("\nProgram Executed Successfully")