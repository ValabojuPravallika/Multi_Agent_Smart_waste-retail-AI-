class SustainabilityAgent:
    """
    Evaluates environmental impact.
    Carbon factor: 0.5 kg CO2 per wasted food unit (WRAP estimate).
    """

    CARBON_FACTOR = 0.5

    def analyze(self, waste_predictions: dict, total_inventory: int) -> tuple:
        total_waste = sum(waste_predictions.values())
        waste_pct = (total_waste / total_inventory * 100) if total_inventory > 0 else 0
        waste_score = round(100 - waste_pct, 1)

        if waste_score >= 80:
            risk_level = "LOW"
        elif waste_score >= 60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        carbon_reduction = round(total_waste * self.CARBON_FACTOR, 1)
        return waste_score, risk_level, carbon_reduction