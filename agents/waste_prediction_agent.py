import pandas as pd


class WastePredictionAgent:
    """Estimates unsold inventory as max(Stock - PredictedSales, 0)."""

    def analyze(self, inventory: pd.DataFrame, demand_forecast: dict) -> dict:
        waste_predictions = {}
        for _, row in inventory.iterrows():
            stock = int(row["Stock"])
            predicted_sales = demand_forecast.get(row["Product"], 0)
            expected_waste = max(stock - predicted_sales, 0)
            waste_predictions[row["Product"]] = expected_waste
        return waste_predictions