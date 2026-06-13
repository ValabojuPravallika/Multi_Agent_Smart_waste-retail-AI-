"""
SmartWaste Retail AI — Full-Stack Multi-Agent Food Waste Reduction Platform
Woosong University · AI Venture Project · 2025/2026

UPGRADE NOTES (Novelty additions):
1. Recommendation Agent now performs a closed-loop negotiation with the
   Sustainability/Discount agents when risk remains HIGH after the first
   pass — escalating discounts and re-checking risk (Agent 7 -> 7b).
2. Explainability Agent now calls an LLM (Claude) to generate a natural
   -language executive briefing from all structured agent outputs, with
   an offline rule-based fallback if the API call fails.
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
[data-testid="stMetricValue"] { font-size:1.5rem; font-weight:600; }
[data-testid="stMetricLabel"] { font-size:0.8rem; color:#888; }
.agent-log-entry { font-family:monospace; font-size:12px; padding:4px 0;
                   border-bottom:1px solid rgba(255,255,255,0.05); }
.agent-log-entry .ts    { color:#4CAF50; margin-right:8px; }
.agent-log-entry .agent { color:#64B5F6; margin-right:6px; }
.agent-log-entry .msg   { color:#E0E0E0; }
.shap-bar-pos { background:linear-gradient(90deg,#1B5E20,#4CAF50); border-radius:3px; height:18px; }
.shap-bar-neg { background:linear-gradient(90deg,#B71C1C,#EF5350); border-radius:3px; height:18px; }
.negotiation-entry { font-family:monospace; font-size:12px; padding:3px 0; color:#FFB74D; }
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
    for a in ["Inventory Agent","Expiry Agent","Demand Forecast Agent",
              "Waste Prediction Agent","Discount Agent","Sustainability Agent",
              "Recommendation Agent (negotiation)","Explainability Agent (LLM)"]:
        st.markdown(
            f'<span style="display:inline-block;background:rgba(76,175,80,0.15);'
            f'color:#4CAF50;border-radius:12px;padding:2px 10px;font-size:12px;margin:2px;">✦ {a}</span>',
            unsafe_allow_html=True
        )
    st.divider()
    st.caption("📊 Dataset: Kaggle Retail · 73,100 rows")
    st.caption("🤖 Model: Random Forest + SHAP XAI")
    st.caption("🧠 Narrative: Claude LLM (with offline fallback)")
    st.caption("Woosong University · 2025/2026")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("♻️ SmartWaste Retail AI")
st.markdown(f"**Store:** {store} &nbsp;|&nbsp; **System:** Multi-Agent Food Waste Reduction Platform with Closed-Loop Negotiation & SHAP/LLM Explainability")
st.divider()

# ─────────────────────────────────────────────
# INVENTORY INPUT
# ─────────────────────────────────────────────
st.subheader("📦 Store Inventory")
st.caption("Edit product data below, then run the analysis.")

DEFAULT_INVENTORY = pd.DataFrame({
    "Product":    ["Sandwich","Milk","Lunch Box","Salad","Juice"],
    "Stock":      [100,50,80,40,70],
    "ExpiryDays": [1,2,1,1,5],
    "Price":      [4.0,2.0,6.0,5.0,3.0],
})

inventory = st.data_editor(
    DEFAULT_INVENTORY, num_rows="dynamic", use_container_width=True,
    column_config={
        "Product":    st.column_config.TextColumn("Product"),
        "Stock":      st.column_config.NumberColumn("Stock (units)", min_value=0),
        "ExpiryDays": st.column_config.NumberColumn("Expiry (days)", min_value=1),
        "Price":      st.column_config.NumberColumn("Price ($)", min_value=0.0, format="$%.2f"),
    },
)
st.divider()

run_col, _ = st.columns([1,3])
with run_col:
    run_clicked = st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)

if run_clicked:
    total_inventory = int(inventory["Stock"].sum())
    agent_log = []

    def log(agent_name, message):
        agent_log.append({"time": time.strftime("%H:%M:%S"), "agent": agent_name, "message": message})

    # ── Agent instances ──────────────────────
    inv_agent      = InventoryAgent()
    exp_agent      = ExpiryAgent()
    demand_agent   = DemandForecastAgent()
    waste_agent    = WastePredictionAgent()
    discount_agent = DiscountAgent()
    sust_agent     = SustainabilityAgent()
    rec_agent      = RecommendationAgent()
    expl_agent     = ExplainabilityAgent()

    # ── Pipeline ─────────────────────────────
    progress = st.progress(0, text="Initializing agents…")

    progress.progress(8, text="Agent 1/9 — Inventory Agent scanning stock…")
    inv_result = inv_agent.analyze(inventory)
    log("Inventory Agent", f"Total stock: {inv_result['total_stock']} units | Products: {inv_result['product_count']} | Low-stock: {inv_result['low_stock_items'] or 'None'}")

    progress.progress(18, text="Agent 2/9 — Expiry Agent detecting near-expiry products…")
    expiring = exp_agent.analyze(inventory)
    critical = len(expiring[expiring["Urgency"]=="Critical"]) if not expiring.empty else 0
    warning  = len(expiring[expiring["Urgency"]=="Warning"])  if not expiring.empty else 0
    log("Expiry Agent", f"Near-expiry: {len(expiring)} | Critical (≤1 day): {critical} | Warning (2 days): {warning}")

    progress.progress(30, text="Agent 3/9 — Demand Forecast Agent training Random Forest + SHAP…")
    demand_forecast = demand_agent.analyze(inventory)
    model_metrics   = demand_agent.get_metrics()
    log("Demand Forecast Agent", f"Kaggle 73,100 rows | MAE: {model_metrics['MAE']} | RMSE: {model_metrics['RMSE']} | R²: {model_metrics['R2']} | " + " | ".join([f"{p}:{v}" for p,v in demand_forecast.items()]))

    progress.progress(40, text="Agent 3/9 — Computing SHAP values for all products…")
    shap_values = demand_agent.get_shap_values(inventory)
    log("Demand Forecast Agent", f"SHAP TreeExplainer computed for {len(shap_values)} products")

    progress.progress(50, text="Agent 4/9 — Waste Prediction Agent estimating unsold units…")
    waste_preds = waste_agent.analyze(inventory, demand_forecast)
    log("Waste Prediction Agent", " | ".join([f"{p}:{v}" for p,v in waste_preds.items()]) + f" | Total: {sum(waste_preds.values())} units")

    progress.progress(58, text="Agent 5/9 — Discount Agent computing tiered pricing…")
    discounts = discount_agent.analyze(waste_preds)
    log("Discount Agent", " | ".join([f"{d['Product']}:{d['Discount']}" for d in discounts]))

    progress.progress(66, text="Agent 6/9 — Sustainability Agent evaluating environmental impact…")
    waste_score, risk_level, carbon_reduction = sust_agent.analyze(waste_preds, total_inventory)
    log("Sustainability Agent", f"Waste Score: {waste_score}% | Risk: {risk_level} | CO₂: {carbon_reduction} kg")

    progress.progress(74, text="Agent 7/9 — Recommendation Agent generating action list…")
    recommendations = rec_agent.analyze(discounts, risk_level)
    log("Recommendation Agent", f"Generated {len(recommendations)} prioritized actions | Risk: {risk_level}")

    # ══════════════════════════════════════
    # NEW: Closed-loop negotiation (Agent 7b)
    # ══════════════════════════════════════
    progress.progress(82, text="Agent 7b/9 — Closed-loop negotiation (Recommendation ↔ Sustainability)…")
    discounts, risk_level, waste_score, carbon_reduction, negotiation_log = rec_agent.negotiate(
        discounts, waste_preds, sust_agent, total_inventory
    )
    for entry in negotiation_log:
        log("Negotiation Loop", entry)
    if any("Escalation" in e for e in negotiation_log):
        recommendations = rec_agent.analyze(discounts, risk_level)
        log("Recommendation Agent", f"Re-generated {len(recommendations)} actions after escalation | Risk: {risk_level}")

    progress.progress(92, text="Agent 8/9 — Explainability Agent generating LLM business brief…")
    revenue_saved = sum(waste_preds[r["Product"]] * r["Price"] for _,r in inventory.iterrows())
    explanation = expl_agent.generate(
        waste_score, risk_level, sum(waste_preds.values()),
        revenue_saved, carbon_reduction,
        mae=model_metrics["MAE"], rmse=model_metrics["RMSE"],
        shap_summary=shap_values,
        recommendations=recommendations,
        negotiation_log=negotiation_log,
    )
    log("Explainability Agent", "LLM-generated business insight report produced (with SHAP + negotiation context).")

    progress.progress(100, text="✅ All 9 pipeline stages completed.")
    time.sleep(0.3)
    progress.empty()

    total_waste_units = sum(waste_preds.values())
    waste_pct = round((total_waste_units / total_inventory) * 100, 1) if total_inventory else 0

    # ════════════════════════════════════════
    # AGENT COMMUNICATION LOG
    # ════════════════════════════════════════
    with st.expander("🔁 Agent Communication Log", expanded=False):
        for entry in agent_log:
            css_class = "negotiation-entry" if entry["agent"] == "Negotiation Loop" else "agent-log-entry"
            st.markdown(
                f'<div class="{css_class}">'
                f'<span class="ts">[{entry["time"]}]</span>'
                f'<span class="agent">{entry["agent"]}</span>'
                f'<span class="msg">{entry["message"]}</span>'
                f'</div>', unsafe_allow_html=True
            )

    st.divider()

    # ════════════════════════════════════════
    # KPI DASHBOARD
    # ════════════════════════════════════════
    st.subheader("📊 Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Stock", f"{inv_result['total_stock']} units")
    c2.metric("Waste Score", f"{waste_score}%",
              delta="Good" if waste_score>=80 else "Needs attention",
              delta_color="normal" if waste_score>=80 else "inverse")
    c3.metric("Risk Level", risk_level)
    c4.metric("Projected Waste", f"{total_waste_units} units")
    c5,c6,c7 = st.columns(3)
    c5.metric("Revenue at Risk ($)", f"${revenue_saved:,.0f}")
    c6.metric("Waste / Total Stock",  f"{waste_pct}%")
    c7.metric("Carbon Reduction (kg CO₂)", f"{carbon_reduction}")
    st.divider()

    # ════════════════════════════════════════
    # ML MODEL PERFORMANCE
    # ════════════════════════════════════════
    st.subheader("🤖 ML Model Performance")
    st.caption(f"Random Forest + SHAP — trained on {model_metrics['Dataset']}")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Model",   "Random Forest")
    m2.metric("Dataset", "73,100 rows")
    m3.metric("Features","7 features")
    m4.metric("MAE",  f"{model_metrics['MAE']} units")
    m5.metric("RMSE", f"{model_metrics['RMSE']} units")
    st.divider()

    # ════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════
    tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
        "📦 Inventory & Expiry",
        "📈 Demand Forecast",
        "🗑️ Waste Prediction",
        "💸 Discount Strategy",
        "🔍 SHAP Explainability",
        "✅ Recommendations",
        "🔄 Negotiation Log",
    ])

    with tab1:
        col_a,col_b = st.columns(2)
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
            x=alt.X("Product:N", sort="-y"), y=alt.Y("Stock:Q", title="Units in Stock"),
            tooltip=["Product","Stock","ExpiryDays","Price"]
        ).properties(height=260, title="Stock Levels by Product")
        st.altair_chart(bar, use_container_width=True)

    with tab2:
        forecast_df = pd.DataFrame(demand_forecast.items(), columns=["Product","Predicted Sales"])
        forecast_df = forecast_df.merge(inventory[["Product","Stock"]], on="Product")
        forecast_df["Fulfillment Rate (%)"] = ((forecast_df["Predicted Sales"]/forecast_df["Stock"])*100).clip(0,100).round(1)
        st.dataframe(forecast_df, use_container_width=True)
        st.caption("Top features: Stock Level (48.9%), Unit Price (26.0%), Category (6.3%), Discount (6.1%)")
        chart = alt.Chart(forecast_df.melt("Product",["Stock","Predicted Sales"],"Metric","Units")).mark_bar().encode(
            x="Product:N", y="Units:Q",
            color=alt.Color("Metric:N", scale=alt.Scale(range=["#81C784","#1B5E20"])),
            xOffset="Metric:N", tooltip=["Product","Metric","Units"]
        ).properties(height=260, title="Stock vs. Predicted Sales")
        st.altair_chart(chart, use_container_width=True)

    with tab3:
        waste_df = pd.DataFrame(waste_preds.items(), columns=["Product","Expected Waste (units)"])
        waste_df["Waste %"] = (waste_df["Expected Waste (units)"]/inventory["Stock"].values*100).round(1)
        st.dataframe(waste_df, use_container_width=True)
        wc = alt.Chart(waste_df).mark_bar().encode(
            x=alt.X("Product:N",sort="-y"), y="Expected Waste (units):Q",
            color=alt.condition(alt.datum["Expected Waste (units)"]>30,alt.value("#c62828"),alt.value("#ef9a9a")),
            tooltip=["Product","Expected Waste (units)","Waste %"]
        ).properties(height=260, title="Expected Waste by Product")
        st.altair_chart(wc, use_container_width=True)

    with tab4:
        discount_df = pd.DataFrame(discounts).drop(columns=["Discount_int"],errors="ignore")
        st.dataframe(discount_df, use_container_width=True)
        dc = alt.Chart(pd.DataFrame(discounts)).mark_bar(color="#388e3c").encode(
            x="Product:N", y=alt.Y("Discount_int:Q",title="Discount (%)"),
            tooltip=["Product","Expected Waste","Discount"]
        ).properties(height=240, title="Recommended Discounts (%)")
        st.altair_chart(dc, use_container_width=True)

    # ══════════════════════════════════════
    # SHAP TAB
    # ══════════════════════════════════════
    with tab5:
        st.markdown("### 🔍 SHAP Explainability — Why did the model predict these sales?")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) shows the **exact contribution of each feature** "
            "to the demand prediction for every product. "
            "**Green bars = feature increases predicted sales. Red bars = feature decreases predicted sales.**"
        )
        st.divider()

        selected_product = st.selectbox(
            "Select a product to inspect:",
            options=list(shap_values.keys())
        )

        sv_data     = shap_values[selected_product]
        shap_dict   = sv_data["shap_values"]
        base_val    = sv_data["base_value"]
        prediction  = sv_data["prediction"]
        waste_units = waste_preds[selected_product]

        sc1,sc2,sc3 = st.columns(3)
        sc1.metric("Base Prediction (avg)",  f"{base_val:.1f} units", help="Average prediction across all training data")
        sc2.metric("Adjusted Prediction",    f"{prediction:.1f} units", help="Final prediction after SHAP adjustments")
        sc3.metric("Expected Waste",         f"{waste_units} units", delta=f"{round((waste_units/inventory[inventory['Product']==selected_product]['Stock'].values[0])*100,1)}% of stock", delta_color="inverse")

        st.markdown(f"#### Feature Contributions for **{selected_product}**")
        st.caption("Each bar shows how much that feature pushed the sales prediction up (green) or down (red) from the baseline.")

        shap_rows = []
        cumulative = base_val
        for feat, val in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True):
            shap_rows.append({
                "Feature":      feat,
                "SHAP Value":   val,
                "Direction":    "Increases sales ↑" if val > 0 else "Decreases sales ↓",
                "Abs":          abs(val),
                "Cumulative":   cumulative + val,
            })
            cumulative += val

        shap_df = pd.DataFrame(shap_rows)

        waterfall = alt.Chart(shap_df).mark_bar(size=32).encode(
            y=alt.Y("Feature:N",
                    sort=alt.EncodingSortField(field="Abs", order="descending"),
                    title=""),
            x=alt.X("SHAP Value:Q",
                    title="SHAP Value (units of predicted sales)",
                    scale=alt.Scale(domainMin=min(shap_df["SHAP Value"].min()-5, -5),
                                    domainMax=max(shap_df["SHAP Value"].max()+5, 5))),
            color=alt.Color("Direction:N",
                            scale=alt.Scale(
                                domain=["Increases sales ↑","Decreases sales ↓"],
                                range=["#2E7D32","#C62828"]
                            )),
            tooltip=["Feature","SHAP Value","Direction"]
        ).properties(height=280, title=f"SHAP Feature Contributions — {selected_product}")

        zero_line = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(color="white", opacity=0.4).encode(x="x:Q")

        st.altair_chart(waterfall + zero_line, use_container_width=True)

        top_pos = [(f,v) for f,v in shap_dict.items() if v > 0]
        top_neg = [(f,v) for f,v in shap_dict.items() if v < 0]
        top_pos.sort(key=lambda x:-x[1])
        top_neg.sort(key=lambda x:x[1])

        st.markdown("**📖 Plain-language interpretation:**")
        interp = f"The model predicts **{prediction:.0f} units** of sales for **{selected_product}** (baseline: {base_val:.0f} units). "
        if top_pos:
            interp += f"The biggest factor **boosting** predicted sales is **{top_pos[0][0]}** (+{top_pos[0][1]:.1f} units). "
        if top_neg:
            interp += f"The biggest factor **reducing** predicted sales is **{top_neg[0][0]}** ({top_neg[0][1]:.1f} units). "
        interp += f"After all adjustments, the model expects **{waste_units} units** to go unsold."
        st.info(interp)

        st.divider()

        st.markdown("#### SHAP Summary — All Products")
        st.caption("The most influential feature for each product's demand prediction.")

        summary_rows = []
        for prod, data in shap_values.items():
            sv = data["shap_values"]
            top_feat = max(sv, key=lambda k: abs(sv[k]))
            summary_rows.append({
                "Product":        prod,
                "Prediction (units)": data["prediction"],
                "Top SHAP Feature":   top_feat,
                "SHAP Impact":        sv[top_feat],
                "Direction":          "↑ Boosts sales" if sv[top_feat]>0 else "↓ Reduces sales",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        fi_rows = []
        for feat in demand_agent.FEATURE_LABELS:
            mean_abs = sum(abs(shap_values[p]["shap_values"].get(feat,0)) for p in shap_values) / len(shap_values)
            fi_rows.append({"Feature": feat, "Mean |SHAP| (units)": round(mean_abs,2)})
        fi_df = pd.DataFrame(fi_rows).sort_values("Mean |SHAP| (units)", ascending=False)

        fi_chart = alt.Chart(fi_df).mark_bar(color="#0277BD").encode(
            y=alt.Y("Feature:N", sort="-x", title=""),
            x=alt.X("Mean |SHAP| (units):Q"),
            tooltip=["Feature","Mean |SHAP| (units)"]
        ).properties(height=220, title="Global Feature Importance (Mean |SHAP| across all products)")
        st.altair_chart(fi_chart, use_container_width=True)

    with tab6:
        for rec in recommendations:
            if "🔴" in rec:   st.error(rec)
            elif "🟡" in rec or "⚠️" in rec: st.warning(rec)
            else: st.success(rec)

    # ══════════════════════════════════════
    # NEW: NEGOTIATION LOG TAB
    # ══════════════════════════════════════
    with tab7:
        st.markdown("### 🔄 Closed-Loop Agent Negotiation")
        st.markdown(
            "When the **Sustainability Agent** reports `HIGH` risk after the first "
            "discount pass, the **Recommendation Agent** automatically negotiates "
            "with the **Discount Agent**: it escalates discounts for the highest-waste "
            "products (≥30 units), then re-checks the risk level. This closed feedback "
            "loop is what distinguishes an *agentic* pipeline from a simple "
            "sequential script."
        )
        st.divider()
        if negotiation_log:
            for entry in negotiation_log:
                if "Escalation" in entry:
                    st.warning(f"🔁 {entry}")
                else:
                    st.info(f"ℹ️ {entry}")
        else:
            st.success("No negotiation needed — risk level acceptable after first pass.")

    st.divider()

    # ════════════════════════════════════════
    # EXPLAINABILITY REPORT
    # ════════════════════════════════════════
    st.subheader("🧠 Explainability Agent — AI-Generated Business Insight Report")
    st.code(explanation, language=None)

    st.divider()

    # ════════════════════════════════════════
    # PIPELINE ARCHITECTURE
    # ════════════════════════════════════════
    st.subheader("🔄 Multi-Agent Pipeline")
    arch_cols = st.columns([1,0.25,1,0.25,1,0.25,1])
    with arch_cols[0]:
        st.markdown("**Observe**")
        st.info("1️⃣ Inventory Agent\n\nTotal stock · low-stock flags")
        st.info("2️⃣ Expiry Agent\n\nCritical & warning products")
    with arch_cols[1]:
        st.markdown(" ")
        st.markdown("<br><br><br><h2 style='text-align:center'>→</h2>", unsafe_allow_html=True)
    with arch_cols[2]:
        st.markdown("**Predict**")
        st.success("3️⃣ Demand Forecast\n\nRandom Forest + **SHAP XAI**\nKaggle 73K rows · 7 features")
        st.success("4️⃣ Waste Prediction\n\nstock − forecast = waste")
    with arch_cols[3]:
        st.markdown(" ")
        st.markdown("<br><br><br><h2 style='text-align:center'>→</h2>", unsafe_allow_html=True)
    with arch_cols[4]:
        st.markdown("**Act & Negotiate**")
        st.warning("5️⃣ Discount Agent\n\nTiered 10–50% pricing")
        st.warning("6️⃣ Sustainability\n\nWaste score · CO₂ impact")
        st.warning("7️⃣ Recommendation\n\n🔁 Closed-loop negotiation")
    with arch_cols[5]:
        st.markdown(" ")
        st.markdown("<br><br><br><h2 style='text-align:center'>→</h2>", unsafe_allow_html=True)
    with arch_cols[6]:
        st.markdown("**Explain**")
        st.error("8️⃣ Explainability\n\nLLM business brief +\nSHAP drivers")
