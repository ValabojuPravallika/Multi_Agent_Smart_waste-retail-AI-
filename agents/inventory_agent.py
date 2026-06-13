import pandas as pd


class InventoryAgent:
    """Monitors stock levels and flags low-stock items."""

    LOW_STOCK_THRESHOLD = 30

    def analyze(self, inventory: pd.DataFrame) -> dict:
        total_stock = int(inventory["Stock"].sum())
        low_stock = inventory[inventory["Stock"] <= self.LOW_STOCK_THRESHOLD]["Product"].tolist()

        return {
            "total_stock": total_stock,
            "low_stock_items": low_stock,
            "product_count": len(inventory)
        }