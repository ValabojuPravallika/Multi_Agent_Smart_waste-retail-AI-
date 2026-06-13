import pandas as pd


class ExpiryAgent:
    """Identifies products nearing expiration (within 2 days)."""

    EXPIRY_THRESHOLD_DAYS = 2

    def analyze(self, inventory: pd.DataFrame) -> pd.DataFrame:
        expiring = inventory[
            inventory["ExpiryDays"] <= self.EXPIRY_THRESHOLD_DAYS
        ].copy()

        expiring["Urgency"] = expiring["ExpiryDays"].apply(
            lambda d: "Critical" if d <= 1 else "Warning"
        )

        return expiring.reset_index(drop=True)