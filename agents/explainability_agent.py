class ExplainabilityAgent:
    """
    Generates a human-readable business insight summary combining
    all upstream agent outputs including SHAP feature attributions.
    """

    def generate(
        self,
        waste_score:       float,
        risk_level:        str,
        total_waste_units: int,
        revenue_saved:     float,
        carbon_reduction:  float,
        mae:               float = None,
        rmse:              float = None,
        shap_summary:      dict  = None,
    ) -> str:

        risk_desc = {
            "LOW":    "The store is managing inventory efficiently with minimal projected waste.",
            "MEDIUM": "Moderate waste risk detected. Targeted discounts are recommended.",
            "HIGH":   "High waste risk. Immediate intervention required to prevent losses."
        }.get(risk_level, "")

        metrics_block = ""
        if mae is not None and rmse is not None:
            metrics_block = f"\n📐 Forecast Model Accuracy (Random Forest + SHAP)\n  MAE: {mae} units  |  RMSE: {rmse} units\n"

        # SHAP top driver block — most impactful feature per product
        shap_block = ""
        if shap_summary:
            shap_block = "\n🔍 SHAP Key Drivers (top factor per product)\n"
            for product, data in shap_summary.items():
                sv = data["shap_values"]
                top_feature = max(sv, key=lambda k: abs(sv[k]))
                direction   = "↑ increases" if sv[top_feature] > 0 else "↓ reduces"
                shap_block += (
                    f"  {product}: {top_feature} {direction} "
                    f"predicted sales by {abs(sv[top_feature]):.1f} units\n"
                )

        return f"""
╔══════════════════════════════════════════╗
   SMARTWASTE AI — ANALYSIS SUMMARY
╚══════════════════════════════════════════╝

🧠 Risk Assessment
  Waste Score    : {waste_score}%
  Risk Level     : {risk_level}
  {risk_desc}

📦 Waste Forecast
  Expected Waste : {total_waste_units} units

💰 Financial Impact
  Revenue at Risk : ${revenue_saved:,.0f}

🌱 Environmental Impact
  Carbon Reduction Score : {carbon_reduction} kg CO₂ eq.
{metrics_block}{shap_block}
💡 Key Recommendation
  Apply dynamic discounts to high-risk products.
  Monitor inventory daily and adjust reorder levels
  based on the ML demand forecast and SHAP drivers.
"""
