import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


class DemandForecastAgent:
    """
    Predicts future sales demand using a Random Forest Regressor
    trained on the Kaggle Retail Store Inventory dataset (73,100 rows).
    Features: Inventory Level, Price, Discount, Holiday/Promotion, Seasonality.
    """

    DATA_PATH = "data/retail_store_inventory.csv"

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.le_season = LabelEncoder()
        self.mae = None
        self.rmse = None
        self.r2 = None
        self._trained = False

    def _load_and_prepare(self):
        df = pd.read_csv(self.DATA_PATH)

        # Encode seasonality
        df["Season_enc"] = self.le_season.fit_transform(df["Seasonality"])

        features = ["Inventory Level", "Price", "Discount", "Holiday/Promotion", "Season_enc"]
        X = df[features]
        y = df["Units Sold"]
        return X, y, features

    def _train(self):
        X, y, _ = self._load_and_prepare()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.mae  = round(mean_absolute_error(y_test, y_pred), 2)
        self.rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        self.r2 = round(1 - ss_res / ss_tot, 3)
        self._trained = True

    def analyze(self, inventory: pd.DataFrame) -> dict:
        if not self._trained:
            self._train()

        forecasts = {}
        for _, row in inventory.iterrows():
            # Map ExpiryDays to Seasonality proxy: 1-2 days → "Summer" (urgency), else "Autumn"
            season_label = "Summer" if row["ExpiryDays"] <= 2 else "Autumn"
            season_enc = self.le_season.transform([season_label])[0] \
                if season_label in self.le_season.classes_ else 0

            X_input = pd.DataFrame([{
                "Inventory Level":    row["Stock"],
                "Price":              row["Price"],
                "Discount":           0,
                "Holiday/Promotion":  0,
                "Season_enc":         season_enc,
            }])
            predicted = int(self.model.predict(X_input)[0])
            forecasts[row["Product"]] = max(predicted, 0)

        return forecasts

    def get_metrics(self) -> dict:
        return {
            "MAE":      self.mae,
            "RMSE":     self.rmse,
            "R2":       self.r2,
            "Model":    "Random Forest (n=100, n_jobs=-1)",
            "Features": "Inventory, Price, Discount, Holiday, Seasonality",
            "Dataset":  "Kaggle Retail Store Inventory (73,100 rows)",
        }
