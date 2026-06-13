import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


class DemandForecastAgent:
    """
    Predicts daily sales demand using a Random Forest Regressor.
    Trained on the full Kaggle Retail Store Inventory dataset (73,100 rows).
    Features: Inventory Level, Price, Discount, Holiday/Promotion,
              Seasonality (encoded), Weather (encoded), Category (encoded).
    """

    DATA_PATH = "data/retail_store_inventory.csv"

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.le_season  = LabelEncoder()
        self.le_weather = LabelEncoder()
        self.le_cat     = LabelEncoder()
        self.mae   = None
        self.rmse  = None
        self.r2    = None
        self.fi    = None          # feature importances dict
        self._trained = False
        self.FEATURES = [
            "Inventory Level", "Price", "Discount",
            "Holiday/Promotion", "Season_enc", "Weather_enc", "Category_enc"
        ]

    def _load_and_prepare(self):
        df = pd.read_csv(self.DATA_PATH)

        # Encode categorical columns using ALL unique values seen in full dataset
        df["Season_enc"]   = self.le_season.fit_transform(df["Seasonality"])
        df["Weather_enc"]  = self.le_weather.fit_transform(df["Weather Condition"])
        df["Category_enc"] = self.le_cat.fit_transform(df["Category"])

        X = df[self.FEATURES]
        y = df["Units Sold"]
        return X, y

    def _train(self):
        X, y = self._load_and_prepare()

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

        self.fi = dict(zip(
            self.FEATURES,
            [round(v, 3) for v in self.model.feature_importances_]
        ))
        self._trained = True

    def analyze(self, inventory: pd.DataFrame) -> dict:
        if not self._trained:
            self._train()

        # Map convenience store product context to dataset encodings
        def get_season(expiry_days):
            # Products expiring soon behave like peak-season items
            label = "Summer" if expiry_days <= 2 else "Autumn"
            classes = list(self.le_season.classes_)
            return self.le_season.transform([label])[0] if label in classes else 0

        def get_weather():
            # Default to Sunny (most common in dataset)
            classes = list(self.le_weather.classes_)
            label = "Sunny" if "Sunny" in classes else classes[0]
            return self.le_weather.transform([label])[0]

        def get_category():
            # Map to Groceries (closest to convenience store food items)
            classes = list(self.le_cat.classes_)
            label = "Groceries" if "Groceries" in classes else classes[0]
            return self.le_cat.transform([label])[0]

        weather_enc  = get_weather()
        category_enc = get_category()

        forecasts = {}
        for _, row in inventory.iterrows():
            X_input = pd.DataFrame([{
                "Inventory Level":   row["Stock"],
                "Price":             row["Price"],
                "Discount":          0,
                "Holiday/Promotion": 0,
                "Season_enc":        get_season(row["ExpiryDays"]),
                "Weather_enc":       weather_enc,
                "Category_enc":      category_enc,
            }])
            predicted = int(self.model.predict(X_input)[0])
            forecasts[row["Product"]] = max(predicted, 0)

        return forecasts

    def get_metrics(self) -> dict:
        return {
            "MAE":      self.mae,
            "RMSE":     self.rmse,
            "R2":       self.r2,
            "Model":    "Random Forest (n=100, 7 features, full dataset)",
            "Features": "Inventory, Price, Discount, Holiday, Season, Weather, Category",
            "Dataset":  "Kaggle Retail Store Inventory (73,100 rows, all used)",
            "Feature_Importances": self.fi,
        }
