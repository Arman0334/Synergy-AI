import math

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def load_data(file_path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['class'] = df['class'].apply(lambda x: 1 if x == "Business" else (2 if x == "First Class" else 0))
    df['source'] = df['source'].apply(lambda x: 1 if x == "Sharjah" else (2 if x == "Abu Dhabi" else 0))
    df = df.join(pd.get_dummies(df.airline, prefix='airline')).drop('airline', axis=1)
    df = df.join(pd.get_dummies(df.depart, prefix='depart')).drop('depart', axis=1)
    df = df.join(pd.get_dummies(df.arrival, prefix='arrival')).drop('arrival', axis=1)
    df = df.join(pd.get_dummies(df.destination, prefix='destination')).drop('destination', axis=1)
    return df


def train_model(df: pd.DataFrame) -> tuple[RandomForestRegressor, pd.DataFrame, pd.DataFrame]:
    x, y = df.drop('price', axis=1), df.price
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    best_params= {
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    }
    regressor = RandomForestRegressor(n_jobs=-1, **best_params)
    regressor.fit(x_train, y_train)
    return regressor, x, y


def evaluate_model(regressor: RandomForestRegressor, x_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    y_pred = regressor.predict(x_test)
    return {
        "r2_score": r2_score(y_test, y_pred),
        "mse_score": mean_squared_error(y_test, y_pred),
        "mae_score": mean_absolute_error(y_test, y_pred),
        "rmse_score": math.sqrt(mean_squared_error(y_test, y_pred))
    }


def save_model(regressor: RandomForestRegressor, file_path: str) -> None:
    print("Saving model to: ", file_path)
    joblib.dump(regressor, file_path)


def main() -> None:
    df = load_data("uaeflights.csv")
    df = preprocess_data(df)
    reg, x, y = train_model(df)
    evaluation = evaluate_model(reg, x, y)
    print(evaluation)
    save_model(reg, "flight_fare_prediction_model_uae.pkl")


if __name__ == "__main__":
    main()
