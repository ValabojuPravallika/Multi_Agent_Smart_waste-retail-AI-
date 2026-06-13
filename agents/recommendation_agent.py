class RecommendationAgent:
    """Generates prioritized business actions based on discounts and risk."""

    def analyze(self, discounts: list, risk_level: str) -> list:
        actions = []

        for item in discounts:
            if item["Discount_int"] >= 30:
                actions.append(
                    f"🔴 {item['Product']} — Apply {item['Discount']} discount immediately "
                    f"({item['Expected Waste']} units at risk)"
                )
            elif item["Discount_int"] >= 20:
                actions.append(
                    f"🟡 {item['Product']} — Apply {item['Discount']} discount "
                    f"({item['Expected Waste']} units at risk)"
                )
            else:
                actions.append(
                    f"🟢 {item['Product']} — Monitor closely "
                    f"({item['Expected Waste']} units, low risk)"
                )

        if risk_level == "HIGH":
            actions.append("⚠️ Reduce tomorrow's order quantity by 20%")
            actions.append("⚠️ Activate clearance promotion for near-expiry products")
        elif risk_level == "MEDIUM":
            actions.append("📊 Review reorder quantities — waste level is moderate")
        else:
            actions.append("✅ Inventory health is good — maintain current strategy")

        return actions