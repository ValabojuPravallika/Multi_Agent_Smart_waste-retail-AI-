"""
SmartWaste Retail AI — Full-Stack Multi-Agent Food Waste Reduction Platform
Woosong University · AI Venture Project · 2025
"""

import streamlit as st
import pandas as pd
import altair as alt
import time

from agents.inventory_agent import InventoryAgent
from agents.expiry_agent import ExpiryAgent
from agents.demand_forecast_agent import DemandForecastAgent
from agents.waste_prediction_agent import WastePredictionAgent
from agents.discount_agent import DiscountAgent
from agents.sustainability_agent import SustainabilityAgent
from agents.recommendation_agent import RecommendationAgent
from agents.explainability_agent import ExplainabilityAgent

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartWaste Retail AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; }
[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #888; }
.agent-log-entry { font-family: monospace; font-size: 12px; padding: 4px 0;
                   border-bottom: 1px solid rgba(255,255,255,0.05); }
.agent-log-entry .ts { color: #4CAF50; margin-right: 8px; }
.agent-log-entry .agent { color: #64B5F6; margin-right: 6px; }
.agent-log-entry .msg { color: #E0E0E0; }
.arch-box { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px; padding: 8px 14px; text-align: center;
            font-size: 12px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/recycle.png", width=60)
    st.title("SmartWaste AI")
    st.caption("Multi-Agent Food Waste Reduction")
    st.divider()

    store = st.selectbox(
        "🏪 Select Store",
        ["GS25 — Daejeon Branch", "CU — Seoul Branch", "7-Eleven — Busan Branch"],
    )

    st.divider()
    st.markdown("**System Agents**")
    for a in [
        "Inventory Agent", "Expiry Agent", "Demand Forecast Agent",
        "Waste Prediction Agent", "Discount Agent",
        "Sustainability Agent", "Recommendation Agent", "Explainability Agent",
    ]:
        st.markdown(
            f'<span style="display:inline-block;background:rgba(76,175,80,0.15);'
            f'color:#4CAF50;border-radius:12px;padding:2px 10px;font-size:12px;margin:2px;">✦ {a}</span>',
            unsafe_allow_html=True
        )
    st.divider()
    st.caption("📊 Dataset: Kaggle Retail Store Inventory · 73,100 rows")
    st.caption("🤖 Model: Random Forest Regressor")
    st.caption("Woosong University · 2025")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("♻️ SmartWaste Retail AI")
st.markdown(f"**Store:** {store} &nbsp;|&nbsp; **System:** Multi-Agent Food Waste Reduction Platform")
st.divider()

# ─────────────────────────────────────────────
# INVENTORY INPUT
# ─────────────────────────────────────────────
st.subheader("📦 Store Inventory")
st.caption("Edit product data below, then run the analysis.")

DEFAULT_INVENTORY = pd.DataFrame({
    "Product":    ["Sandwich", "Milk", "Lunch Box", "Salad", "Juice"],
    "Stock":      [100, 50, 80, 40, 70],
    "ExpiryDays": [1, 2, 1, 1, 5],
    "Price":      [4.0, 2.0, 6.0, 5.0, 3.0],
})

inventory = st.data_editor(
    DEFAULT_INVENTORY,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Product":    st.column_config.TextColumn("Product"),
        "Stock":      st.column_config.NumberColumn("Stock (units)", min_value=0),
        "ExpiryDays": st.column_config.NumberColumn("Expiry (days)", min_value=1),
        "Price":      st.column_config.NumberColumn("Price ($)", min_value=0.0, format="$%.2f"),
    },
)

st.divider()

# ─────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    run_clicked = st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)

if run_clicked:
    total_inventory = int(inventory["Stock"].sum())
    agent_log = []

    def log(agent_name, message):
        ts = time.strftime("%H:%M:%S")
        agent_log.append({"time": ts, "agent": agent_name, "message": message})

    # ── Agents ──────────────────────────────
    inv_agent      = InventoryAgent()
    exp_agent      = ExpiryAgent()
    demand_agent   = DemandForecastAgent()
    waste_agent    = WastePredictionAgent()
    discount_agent = DiscountAgent()
    sust_agent     = SustainabilityAgent()
    rec_agent      = RecommendationAgent()
    expl_agent     = ExplainabilityAgent()

    # ── Pipeline ────────────────────────────
    progress = st.progress(0, text="Initializing agents…")

    progress.progress(10, text="Agent 1/8 — Inventory Agent scanning stock…")
    inv_result = inv_agent.analyze(inventory)
    log("Inventory Agent", f"Total stock: {inv_result['total_stock']} units | Products: {inv_result['product_count']} | Low-stock items: {inv_result['low_stock_items'] or 'None'}")

    progress.progress(22, text="Agent 2/8 — Expiry Agent detecting near-expiry products…")
    expiring = exp_agent.analyze(inventory)
    critical = len(expiring[expiring["Urgency"] == "Critical"]) if not expiring.empty else 0
    warning  = len(expiring[expiring["Urgency"] == "Warning"])  if not expiring.empty else 0
    log("Expiry Agent", f"Near-expiry products: {len(expiring)} | Critical (≤1 day): {critical} | Warning (2 days): {warning}")

    progress.progress(35, text="Agent 3/8 — Demand Forecast Agent training on Kaggle dataset (73,100 rows)…")
    demand_forecast = demand_agent.analyze(inventory)
    model_metrics   = demand_agent.get_metrics()
    forecast_summary = " | ".join([f"{p}: {v}" for p, v in demand_forecast.items()])
    log("Demand Forecast Agent", f"Trained on {model_metrics['Dataset']} | MAE: {model_metrics['MAE']} | RMSE: {model_metrics['RMSE']} | R²: {model_metrics['R2']} | Forecasts → {forecast_summary}")

    progress.progress(50, text="Agent 4/8 — Waste Prediction Agent estimating unsold units…")
    waste_preds = waste_agent.analyze(inventory, demand_forecast)
    waste_summary = " | ".join([f"{p}: {v}" for p, v in waste_preds.items()])
    log("Waste Prediction Agent", f"Predicted waste per product → {waste_summary} | Total: {sum(waste_preds.values())} units")

    progress.progress(62, text="Agent 5/8 — Discount Agent computing tiered pricing…")
    discounts = discount_agent.analyze(waste_preds)
    disc_summary = " | ".join([f"{d['Product']}: {d['Discount']}" for d in discounts])
    log("Discount Agent", f"Discount recommendations → {disc_summary}")

    progress.progress(75, text="Agent 6/8 — Sustainability Agent evaluating environmental impact…")
    waste_score, risk_level, carbon_reduction = sust_agent.analyze(waste_preds, total_inventory)
    log("Sustainability Agent", f"Waste Score: {waste_score}% | Risk Level: {risk_level} | Carbon Reduction: {carbon_reduction} kg CO₂ eq.")

    progress.progress(87, text="Agent 7/8 — Recommendation Agent generating action list…")
    recommendations = rec_agent.analyze(discounts, risk_level)
    log("Recommendation Agent", f"Generated {len(recommendations)} prioritized actions | Risk level: {risk_level}")

    progress.progress(95, text="Agent 8/8 — Explainability Agent generating business report…")
    explanation = expl_agent.generate(
        waste_score, risk_level, sum(waste_preds.values()),
        sum(waste_preds[r["Product"]] * r["Price"] for _, r in inventory.iterrows()),
        carbon_reduction, mae=model_metrics["MAE"], rmse=model_metrics["RMSE"]
    )
    log("Explainability Agent", "Business insight report generated successfully.")

    progress.progress(100, text="✅ All 8 agents completed.")
    time.sleep(0.4)
    progress.empty()

    # ── KPI Calculations ────────────────────
    total_waste_units = sum(waste_preds.values())
    revenue_saved     = sum(waste_preds[r["Product"]] * r["Price"] for _, r in inventory.iterrows())
    waste_pct         = round((total_waste_units / total_inventory) * 100, 1) if total_inventory else 0

    # ════════════════════════════════════════
    # AGENT COMMUNICATION LOG
    # ════════════════════════════════════════
    with st.expander("🔁 Agent Communication Log — see what each agent received and output", expanded=True):
        for entry in agent_log:
            st.markdown(
                f'<div class="agent-log-entry">'
                f'<span class="ts">[{entry["time"]}]</span>'
                f'<span class="agent">{entry["agent"]}</span>'
                f'<span class="msg">{entry["message"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # ════════════════════════════════════════
    # KPI DASHBOARD
    # ════════════════════════════════════════
    st.subheader("📊 Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Stock", f"{inv_result['total_stock']} units")
    c2.metric("Waste Score", f"{waste_score}%",
              delta="Good" if waste_score >= 80 else "Needs attention",
              delta_color="normal" if waste_score >= 80 else "inverse")
    c3.metric("Risk Level", risk_level)
    c4.metric("Projected Waste", f"{total_waste_units} units")

    c5, c6, c7 = st.columns(3)
    c5.metric("Revenue at Risk ($)", f"${revenue_saved:,.0f}")
    c6.metric("Waste / Total Stock",  f"{waste_pct}%")
    c7.metric("Carbon Reduction (kg CO₂)", f"{carbon_reduction}")

    st.divider()

    # ════════════════════════════════════════
    # ML MODEL PERFORMANCE
    # ════════════════════════════════════════
    st.subheader("🤖 ML Model Performance")
    st.caption(f"Random Forest — trained on {model_metrics['Dataset']}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Model",    "Random Forest")
    m2.metric("Dataset",  "73,100 rows")
    m3.metric("Features", "5 features")
    m4.metric("MAE",  f"{model_metrics['MAE']} units")
    m5.metric("RMSE", f"{model_metrics['RMSE']} units")

    st.divider()

    # ════════════════════════════════════════
    # DETAIL TABS
    # ════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📦 Inventory & Expiry",
        "📈 Demand Forecast",
        "🗑️ Waste Prediction",
        "💸 Discount Strategy",
        "✅ Recommendations",
    ])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Full Inventory**")
            st.dataframe(inventory, use_container_width=True)
        with col_b:
            st.markdown("**Products Near Expiry (≤ 2 days)**")
            if not expiring.empty:
                st.dataframe(expiring, use_container_width=True)
            else:
                st.success("No products expiring within 2 days.")

        bar = alt.Chart(inventory).mark_bar(color="#4CAF50").encode(
            x=alt.X("Product:N", sort="-y", title="Product"),
            y=alt.Y("Stock:Q",   title="Units in Stock"),
            tooltip=["Product", "Stock", "ExpiryDays", "Price"]
        ).properties(height=260, title="Stock Levels by Product")
        st.altair_chart(bar, use_container_width=True)

    with tab2:
        forecast_df = pd.DataFrame(demand_forecast.items(), columns=["Product", "Predicted Sales"])
        forecast_df = forecast_df.merge(inventory[["Product", "Stock"]], on="Product")
        forecast_df["Fulfillment Rate (%)"] = (
            (forecast_df["Predicted Sales"] / forecast_df["Stock"]) * 100
        ).clip(0, 100).round(1)
        st.dataframe(forecast_df, use_container_width=True)

        st.caption(f"Model trained on real Kaggle retail data · Top features: Inventory Level (50.7%), Price (33.4%), Discount (6.9%)")

        chart = alt.Chart(
            forecast_df.melt("Product", ["Stock", "Predicted Sales"], "Metric", "Units")
        ).mark_bar().encode(
            x=alt.X("Product:N"),
            y=alt.Y("Units:Q"),
            color=alt.Color("Metric:N", scale=alt.Scale(range=["#81C784", "#1B5E20"])),
            xOffset="Metric:N",
            tooltip=["Product", "Metric", "Units"]
        ).properties(height=260, title="Stock vs. Predicted Sales")
        st.altair_chart(chart, use_container_width=True)

    with tab3:
        waste_df = pd.DataFrame(waste_preds.items(), columns=["Product", "Expected Waste (units)"])
        waste_df["Waste %"] = (waste_df["Expected Waste (units)"] / inventory["Stock"].values * 100).round(1)
        st.dataframe(waste_df, use_container_width=True)

        waste_chart = alt.Chart(waste_df).mark_bar().encode(
            x=alt.X("Product:N", sort="-y"),
            y=alt.Y("Expected Waste (units):Q"),
            color=alt.condition(
                alt.datum["Expected Waste (units)"] > 30,
                alt.value("#c62828"), alt.value("#ef9a9a")
            ),
            tooltip=["Product", "Expected Waste (units)", "Waste %"]
        ).properties(height=260, title="Expected Waste by Product")
        st.altair_chart(waste_chart, use_container_width=True)

    with tab4:
        discount_df = pd.DataFrame(discounts).drop(columns=["Discount_int"], errors="ignore")
        st.dataframe(discount_df, use_container_width=True)

        disc_chart = alt.Chart(pd.DataFrame(discounts)).mark_bar(color="#388e3c").encode(
            x=alt.X("Product:N"),
            y=alt.Y("Discount_int:Q", title="Discount (%)"),
            tooltip=["Product", "Expected Waste", "Discount"]
        ).properties(height=240, title="Recommended Discounts (%)")
        st.altair_chart(disc_chart, use_container_width=True)

    with tab5:
        for rec in recommendations:
            if "🔴" in rec:
                st.error(rec)
            elif "🟡" in rec or "⚠️" in rec:
                st.warning(rec)
            else:
                st.success(rec)

    st.divider()

    # ════════════════════════════════════════
    # EXPLAINABILITY REPORT
    # ════════════════════════════════════════
    st.subheader("🧠 Explainability Agent — Business Insight Report")
    st.code(explanation, language=None)

    st.divider()

    # ════════════════════════════════════════
    # ARCHITECTURE DIAGRAM
    # ════════════════════════════════════════
    st.subheader("🔄 Multi-Agent Pipeline")

    arch_cols = st.columns([1, 0.3, 1, 0.3, 1, 0.3, 1])

    with arch_cols[0]:
        st.markdown("**Observe**")
        st.info("1️⃣ Inventory Agent\n\nTotal stock · low-stock flags")
        st.info("2️⃣ Expiry Agent\n\nCritical & warning products")

    with arch_cols[1]:
        st.markdown(" ")
        st.markdown("<br><br><br><h2 style='text-align:center;'>→</h2>", unsafe_allow_html=True)

    with arch_cols[2]:
        st.markdown("**Predict**")
        st.success("3️⃣ Demand Forecast\n\nRandom Forest · Kaggle 73K rows")
        st.success("4️⃣ Waste Prediction\n\nstock − forecast = waste")

    with arch_cols[3]:
        st.markdown(" ")
        st.markdown("<br><br><br><h2 style='text-align:center;'>→</h2>", unsafe_allow_html=True)

    with arch_cols[4]:
        st.markdown("**Act**")
        st.warning("5️⃣ Discount Agent\n\nTiered 10–50% pricing")
        st.warning("6️⃣ Sustainability\n\nWaste score · CO₂ impact")

    with arch_cols[5]:
        st.markdown(" ")
        st.markdown("<br><br><br><h2 style='text-align:center;'>→</h2>", unsafe_allow_html=True)

    with arch_cols[6]:
        st.markdown("**Explain**")
        st.error("7️⃣ Recommendation\n\nPrioritized actions")
        st.error("8️⃣ Explainability\n\nBusiness insight report")
