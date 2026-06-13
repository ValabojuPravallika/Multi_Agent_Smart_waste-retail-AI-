class DiscountAgent:
    """Recommends dynamic discounts based on expected waste volume."""

    TIERS = [
        (5,  10),
        (15, 20),
        (30, 30),
        (50, 40),
        (float("inf"), 50),
    ]

    def analyze(self, waste_predictions: dict) -> list:
        recommendations = []
        for product, expected_waste in waste_predictions.items():
            discount = next(
                pct for threshold, pct in self.TIERS
                if expected_waste <= threshold
            )
            recommendations.append({
                "Product": product,
                "Expected Waste": expected_waste,
                "Discount": f"{discount}%",
                "Discount_int": discount
            })
        return recommendations