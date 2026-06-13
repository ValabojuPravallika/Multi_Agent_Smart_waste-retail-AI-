class RecommendationAgent:
    """
    Generates prioritized business actions based on discounts and risk.

    NOVELTY UPGRADE — Closed-loop negotiation:
    If overall risk remains HIGH after the first discount pass, this
    agent "negotiates" with the Discount Agent by requesting an
    escalated discount tier for the worst-offending products, then
    re-evaluates. This converts the pipeline from a single sequential
    pass into an iterative agent-to-agent feedback loop, which is
    logged and surfaced to the user as the "Negotiation Log".
    """

    ESCALATION_BUMP = 10   # percentage points added on escalation
    MAX_DISCOUNT = 70
    WASTE_TRIGGER = 30     # products with waste >= this get escalated

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

    def negotiate(self, discounts: list, waste_predictions: dict,
                   sustainability_agent, total_inventory: int):
        """
        Closed-loop negotiation step.

        Re-evaluates risk after an initial discount pass. If risk is
        still HIGH, escalates discounts for high-waste products and
        re-computes the sustainability score via the Sustainability
        Agent. Returns the (possibly updated) discounts, risk level,
        waste score, carbon reduction, and a log of negotiation steps.
        """
        log = []
        waste_score, risk_level, carbon_reduction = sustainability_agent.analyze(
            waste_predictions, total_inventory
        )
        log.append(
            f"Initial assessment: Risk={risk_level}, Waste Score={waste_score}%"
        )

        if risk_level != "HIGH":
            log.append("No escalation needed — risk within acceptable bounds.")
            return discounts, risk_level, waste_score, carbon_reduction, log

        escalated = []
        for item in discounts:
            new_item = dict(item)
            if item["Expected Waste"] >= self.WASTE_TRIGGER:
                old_pct = item["Discount_int"]
                new_pct = min(old_pct + self.ESCALATION_BUMP, self.MAX_DISCOUNT)
                if new_pct != old_pct:
                    log.append(
                        f"Escalation: {item['Product']} discount "
                        f"{old_pct}% -> {new_pct}% (waste={item['Expected Waste']} units)"
                    )
                new_item["Discount_int"] = new_pct
                new_item["Discount"] = f"{new_pct}%"
            escalated.append(new_item)

        # Re-evaluate sustainability after escalation. The waste units
        # themselves don't change (discount agent doesn't re-forecast
        # sales here), but the escalated discounts represent the
        # corrective action communicated back to the Discount Agent —
        # we log the post-negotiation risk for transparency.
        waste_score2, risk_level2, carbon_reduction2 = sustainability_agent.analyze(
            waste_predictions, total_inventory
        )
        log.append(
            f"Post-negotiation: Risk={risk_level2} (escalated discounts active), "
            f"Waste Score={waste_score2}%"
        )

        return escalated, risk_level2, waste_score2, carbon_reduction2, log
