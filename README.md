# ♻️ SmartWaste Retail AI

**Multi-Agent AI System for Food Waste Reduction in Convenience Retail**

AI Venture Project · 2026


## ABSTRACT
SmartWaste Retail AI is a full-stack, multi-agent AI system that helps convenience store managers reduce food waste by autonomously observing inventory, predicting demand using machine learning, negotiating pricing actions through a closed feedback loop, and explaining decisions in plain language.

The system is built on a **9-stage agent pipeline** trained on the [Kaggle Retail Store Inventory dataset](https://www.kaggle.com/) (73,100 rows) and deployed on Streamlit Cloud.

---

## 🚀 Live Demo

🔗 **[smartwaste-retail-ai.streamlit.app](https://smartwaste-retai.streamlit.app/)**

---

## ✨ Key Features & Novelty

| Feature | Description |
|---|---|
| **9-Agent Pipeline** | Modular Observe → Predict → Act/Negotiate → Explain architecture |
| **ML Demand Forecasting** | Random Forest Regressor trained on 73,100-row Kaggle dataset (7 features) |
| **SHAP Explainability** | Per-product, per-feature attribution via TreeExplainer (XAI) |
| **🆕 Closed-Loop Negotiation** | Recommendation Agent escalates discounts and re-checks risk with Sustainability Agent when risk stays HIGH — true agent-to-agent feedback loop |
| **🆕 LLM Explainability Agent** | Calls Claude LLM to generate a natural-language executive briefing (with offline rule-based fallback) |
| **Dynamic Discount Tiering** | Tiered pricing 10–60% based on expected waste volume |
| **Full Audit Trail** | Agent Communication Log + Negotiation Log tab for complete traceability |

---

## 🏗️ System Architecture

```
Inventory Input (Streamlit UI)         Kaggle Dataset (73,100 rows)
        │                                        │
        ▼                                        ▼
┌─────────────────────────────────────────────────────┐
│                   OBSERVE PHASE                     │
│  1. Inventory Agent    │   2. Expiry Agent          │
│  Stock totals,         │   Critical / Warning       │
│  low-stock flags       │   near-expiry products     │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                   PREDICT PHASE                     │
│  3. Demand Forecast Agent (Random Forest + SHAP)    │
│  4. Waste Prediction Agent (stock − forecast)       │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              ACT & NEGOTIATE PHASE                  │
│  5. Discount Agent   (tiered 10–60% pricing)        │
│  6. Sustainability Agent (waste score, CO2, risk)   │
│  7. Recommendation Agent                            │
│  7b. 🔁 Closed-Loop Negotiation ──────────────────┐ │
│       if risk = HIGH, escalate discounts           │ │
│       re-check with Sustainability Agent ◄─────────┘ │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                   EXPLAIN PHASE                     │
│  8. Explainability Agent                            │
│     SHAP key drivers + Claude LLM business brief   │
│     (offline rule-based fallback if API unavailable)│
└─────────────────────────────────────────────────────┘
        │
        ▼
  Streamlit Dashboard
  (KPIs · Charts · SHAP Tab · Negotiation Log · AI Brief)
```

---

## 📁 Project Structure

```
smartwaste-retail-ai/
│
├── app.py                          # Main Streamlit application
│
├── agents/
│   ├── inventory_agent.py          # Agent 1 — stock monitoring
│   ├── expiry_agent.py             # Agent 2 — near-expiry detection
│   ├── demand_forecast_agent.py    # Agent 3 — Random Forest + SHAP
│   ├── waste_prediction_agent.py   # Agent 4 — waste estimation
│   ├── discount_agent.py           # Agent 5 — tiered discount pricing
│   ├── sustainability_agent.py     # Agent 6 — waste score + CO2
│   ├── recommendation_agent.py     # Agent 7 — actions + negotiation loop
│   └── explainability_agent.py     # Agent 8 — LLM brief + fallback
│
├── data/
│   └── retail_store_inventory.csv  # Kaggle dataset (73,100 rows)
│
├── requirements.txt
└── README.md
```

---

## 📊 Results (Actual System Run)

| Metric | Value |
|---|---|
| Total Stock | 340 units |
| Waste Score | 48.2% |
| Risk Level | HIGH |
| Projected Waste | 176 units |
| Revenue at Risk | $695 |
| CO2 Reduction Potential | 88.0 kg CO2 eq. |
| Model MAE | 70.3 units |
| Model RMSE | 91.08 units |
| Dataset | Kaggle Retail (73,100 rows) |

### Negotiation Log (HIGH-risk scenario)
```
Initial assessment: Risk=HIGH, Waste Score=48.2%
Escalation: Sandwich discount 50% → 60% (waste=62 units)
Escalation: Lunch Box discount 40% → 50% (waste=34 units)
Escalation: Juice discount 40% → 50% (waste=44 units)
Post-negotiation: Risk=HIGH (escalated discounts active)
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Visualization | Altair (Vega-Lite) |
| Machine Learning | scikit-learn RandomForestRegressor |
| Explainability (XAI) | SHAP TreeExplainer |
| Generative AI | Claude LLM (Anthropic API) — with offline fallback |
| Data | Kaggle Retail Store Inventory (73,100 rows) |
| Language | Python 3 |
| Deployment | Streamlit Community Cloud |

---

## ⚙️ Installation & Local Run

```bash
# 1. Clone the repository
git clone https://github.com/ValabojuPravallika/Multi_Agent_Smart_waste-retail-AI-.git
cd smartwaste-retail-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Add Anthropic API key for LLM narrative
# Create .streamlit/secrets.toml and add:
# ANTHROPIC_API_KEY = "sk-ant-..."

# 4. Run the app
streamlit run app.py
```

---

## 🔑 LLM Configuration (Optional)

The Explainability Agent calls the Claude API to generate a natural-language executive briefing. If no API key is configured, the system automatically falls back to a rule-based summary — the app is fully functional either way.

To enable the LLM narrative:

**Streamlit Cloud:** Go to Manage App → Secrets and add:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

**Local:** Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## 📚 References

1. Kaggle — Retail Store Inventory Forecasting Dataset (73,100 rows)
2. Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP) — NeurIPS
3. WRAP (2021). *Food surplus and waste in the UK* — carbon factor: 0.5 kg CO2 per wasted food unit
4. Anthropic — Claude API Documentation: https://docs.claude.com

---
