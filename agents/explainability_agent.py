import json
import urllib.request


class ExplainabilityAgent:
    """
    Generates a human-readable business insight summary combining
    all upstream agent outputs including SHAP feature attributions.

    NOVELTY UPGRADE:
    Instead of a hard-coded f-string template, this agent now calls an
    LLM (Claude) to synthesize all numeric/structured agent outputs into
    a natural-language executive narrative — turning this from a static
    report generator into a true reasoning/explanation agent.

    If the LLM call fails (no network / no API key), it falls back to
    the original rule-based template so the system remains demoable
    offline.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    MODEL = "claude-sonnet-4-6"

    def generate(
        self,
        waste_score: float,
        risk_level: str,
        total_waste_units: int,
        revenue_saved: float,
        carbon_reduction: float,
        mae: float = None,
        rmse: float = None,
        shap_summary: dict = None,
        recommendations: list = None,
        negotiation_log: list = None,
    ) -> str:

        llm_text = self._try_llm_summary(
            waste_score, risk_level, total_waste_units, revenue_saved,
            carbon_reduction, mae, rmse, shap_summary, recommendations,
            negotiation_log,
        )
        if llm_text:
            return llm_text

        return self._fallback_summary(
            waste_score, risk_level, total_waste_units, revenue_saved,
            carbon_reduction, mae, rmse, shap_summary, negotiation_log,
        )

    # ------------------------------------------------------------------
    # LLM-based narrative generation
    # ------------------------------------------------------------------
    def _try_llm_summary(self, waste_score, risk_level, total_waste_units,
                          revenue_saved, carbon_reduction, mae, rmse,
                          shap_summary, recommendations, negotiation_log):
        try:
            shap_lines = []
            if shap_summary:
                for product, data in shap_summary.items():
                    sv = data["shap_values"]
                    top_feature = max(sv, key=lambda k: abs(sv[k]))
                    direction = "increases" if sv[top_feature] > 0 else "decreases"
                    shap_lines.append(
                        f"{product}: top driver = {top_feature} "
                        f"({direction} predicted sales by {abs(sv[top_feature]):.1f} units)"
                    )

            payload_text = f"""You are the Explainability Agent in a retail food-waste
multi-agent AI system called SmartWaste AI. Write a concise executive
business briefing (120-180 words) for a store manager based on the
following pipeline outputs. Be direct, action-oriented, and avoid
restating raw numbers verbatim where a plain-language interpretation
is clearer.

Waste Score: {waste_score}%
Risk Level: {risk_level}
Total Predicted Waste: {total_waste_units} units
Revenue at Risk: ${revenue_saved:,.0f}
Carbon Reduction Potential: {carbon_reduction} kg CO2 eq.
Forecast Model Accuracy: MAE={mae}, RMSE={rmse}
SHAP Top Drivers:
{chr(10).join(shap_lines) if shap_lines else "N/A"}

Recommended Actions:
{chr(10).join(recommendations) if recommendations else "N/A"}

Negotiation Log (closed-loop agent adjustments, if any):
{chr(10).join(negotiation_log) if negotiation_log else "None"}

Write the briefing as plain prose with short paragraphs. Do not use
markdown headers."""

            body = json.dumps({
                "model": self.MODEL,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": payload_text}],
            }).encode("utf-8")

            req = urllib.request.Request(
                self.API_URL, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            text_parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
            narrative = "\n".join(text_parts).strip()
            if not narrative:
                return None

            header = (
                "============================================\n"
                "  SMARTWASTE AI — AI-GENERATED EXECUTIVE BRIEF\n"
                "============================================\n\n"
            )
            return header + narrative

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Offline fallback (original rule-based template)
    # ------------------------------------------------------------------
    def _fallback_summary(self, waste_score, risk_level, total_waste_units,
                           revenue_saved, carbon_reduction, mae, rmse,
                           shap_summary, negotiation_log):

        risk_desc = {
            "LOW":    "The store is managing inventory efficiently with minimal projected waste.",
            "MEDIUM": "Moderate waste risk detected. Targeted discounts are recommended.",
            "HIGH":   "High waste risk. Immediate intervention required to prevent losses."
        }.get(risk_level, "")

        metrics_block = ""
        if mae is not None and rmse is not None:
            metrics_block = f"\nForecast Model Accuracy (Random Forest + SHAP)\n  MAE: {mae} units  |  RMSE: {rmse} units\n"

        shap_block = ""
        if shap_summary:
            shap_block = "\nSHAP Key Drivers (top factor per product)\n"
            for product, data in shap_summary.items():
                sv = data["shap_values"]
                top_feature = max(sv, key=lambda k: abs(sv[k]))
                direction = "increases" if sv[top_feature] > 0 else "reduces"
                shap_block += (
                    f"  {product}: {top_feature} {direction} "
                    f"predicted sales by {abs(sv[top_feature]):.1f} units\n"
                )

        negotiation_block = ""
        if negotiation_log:
            negotiation_block = "\nClosed-Loop Negotiation Log\n"
            for entry in negotiation_log:
                negotiation_block += f"  - {entry}\n"

        return f"""
============================================
  SMARTWASTE AI — ANALYSIS SUMMARY (offline)
============================================

Risk Assessment
  Waste Score    : {waste_score}%
  Risk Level     : {risk_level}
  {risk_desc}

Waste Forecast
  Expected Waste : {total_waste_units} units

Financial Impact
  Revenue at Risk : ${revenue_saved:,.0f}

Environmental Impact
  Carbon Reduction Score : {carbon_reduction} kg CO2 eq.
{metrics_block}{shap_block}{negotiation_block}
Key Recommendation
  Apply dynamic discounts to high-risk products.
  Monitor inventory daily and adjust reorder levels
  based on the ML demand forecast and SHAP drivers.
"""
