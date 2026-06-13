import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


class DemandForecastAgent:
    """
    Predicts daily sales demand using a Random Forest Regressor.
    Trained on Kaggle Retail Store Inventory dataset (73,100 rows).
    Features: Inventory Level, Price, Discount, Holiday/Promotion,
              Seasonality, Weather Condition, Category.
    Includes SHAP TreeExplainer for per-product feature attribution.
    """

    DATA_PATH = "data/retail_store_inventory.csv"
    # Path is relative to the project root when running via `streamlit run app.py`

    def __init__(self):
        self.model       = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.le_season   = LabelEncoder()
        self.le_weather  = LabelEncoder()
        self.le_cat      = LabelEncoder()
        self.explainer   = None          # SHAP TreeExplainer — set after training
        self.mae         = None
        self.rmse        = None
        self.r2          = None
        self.fi          = None
        self._trained    = False
        self.FEATURES    = [
            "Inventory Level", "Price", "Discount",
            "Holiday/Promotion", "Season_enc", "Weather_enc", "Category_enc"
        ]
        self.FEATURE_LABELS = [
            "Stock Level", "Unit Price", "Discount %",
            "Holiday/Promo", "Seasonality", "Weather", "Category"
        ]

    def _load_and_prepare(self):
        df = pd.read_csv(self.DATA_PATH)
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

        # Build SHAP explainer — no background data (avoids memory issues)
        self.explainer = shap.TreeExplainer(self.model)

        y_pred = self.model.predict(X_test)
        self.mae  = round(mean_absolute_error(y_test, y_pred), 2)
        self.rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        ss_res    = np.sum((y_test - y_pred) ** 2)
        ss_tot    = np.sum((y_test - y_test.mean()) ** 2)
        self.r2   = round(1 - ss_res / ss_tot, 3)
        self.fi   = dict(zip(
            self.FEATURE_LABELS,
            [round(v, 3) for v in self.model.feature_importances_]
        ))
        self._trained = True

    def _build_input(self, row):
        season_label = "Summer" if row["ExpiryDays"] <= 2 else "Autumn"
        classes = list(self.le_season.classes_)
        season_enc  = self.le_season.transform([season_label])[0] if season_label in classes else 0
        weather_enc = self.le_weather.transform(["Sunny"])[0] if "Sunny" in self.le_weather.classes_ else 0
        cat_enc     = self.le_cat.transform(["Groceries"])[0] if "Groceries" in self.le_cat.classes_ else 0
        return pd.DataFrame([{
            "Inventory Level":   row["Stock"],
            "Price":             row["Price"],
            "Discount":          0,
            "Holiday/Promotion": 0,
            "Season_enc":        season_enc,
            "Weather_enc":       weather_enc,
            "Category_enc":      cat_enc,
        }])

    def analyze(self, inventory: pd.DataFrame) -> dict:
        if not self._trained:
            self._train()
        forecasts = {}
        for _, row in inventory.iterrows():
            X_input = self._build_input(row)
            predicted = int(self.model.predict(X_input)[0])
            forecasts[row["Product"]] = max(predicted, 0)
        return forecasts

    def get_shap_values(self, inventory: pd.DataFrame) -> dict:
        """
        Returns per-product SHAP values for each feature.
        Format: { product_name: { feature_label: shap_value, ... }, ... }
        Also includes base_value (expected model output).
        """
        if not self._trained:
            self._train()

        results = {}
        for _, row in inventory.iterrows():
            X_input = self._build_input(row)
            sv = self.explainer.shap_values(X_input)
            shap_dict = dict(zip(self.FEATURE_LABELS, [round(float(v), 2) for v in sv[0]]))
            results[row["Product"]] = {
                "shap_values":  shap_dict,
                "base_value":   round(float(self.explainer.expected_value[0] if hasattr(self.explainer.expected_value,"__len__") else self.explainer.expected_value), 2),
                "prediction":   round(float(self.model.predict(X_input)[0]), 1),
            }
        return results

    def get_metrics(self) -> dict:
        return {
            "MAE":      self.mae,
            "RMSE":     self.rmse,
            "R2":       self.r2,
            "Model":    "Random Forest (n=100, 7 features)",
            "Features": "Inventory, Price, Discount, Holiday, Season, Weather, Category",
            "Dataset":  "Kaggle Retail Store Inventory (73,100 rows, all used)",
            "Feature_Importances": self.fi,
        }
