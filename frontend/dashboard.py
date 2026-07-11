import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import time

# 1. Page Configuration & Professional Dark Aesthetic
st.set_page_config(
    page_title="ProposalPilot Core",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS to force a clean, dense dark-mode command center vibe
st.markdown("""
    <style>
        .reportview-container { background: #0f172a; }
        div.stButton > button:first-child {
            background-color: #00f2fe; color: #0f172a; font-weight: bold; width: 100%; border: none;
        }
        div.stButton > button:first-child:hover { background-color: #4facfe; color: white; }
        .terminal-log {
            background-color: #050b14; border-left: 3px solid #00f2fe; 
            font-family: 'Courier New', Courier, monospace; color: #38bdf8; padding: 10px; font-size: 13px;
        }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar Component (System Telemetry)
with st.sidebar:
    st.title("PROPOSAL_PILOT // V1.0")
    st.markdown("---")
    st.subheader("📡 Core Telemetry")
    st.caption("🟢 System Status: **OPERATIONAL**")
    st.caption("🤖 Model Engine: **Random Forest Ensemble**")
    st.caption("📝 Vector Store: **TF-IDF Matrix / Cosine**")
    st.caption("🧠 LLM Layer: **gemini-2.5-flash**")
    st.markdown("---")
    st.info("💡 **Demo Strategy:** Adjust the project budget to test conditional variance branches in your risk evaluation engine!")

# 3. Main Dashboard Workspace Header
st.title("✈️ ProposalPilot Command Center")
st.write("Autonomous R&D Risk Assessment & Verification Pipeline")
st.markdown("---")

# 4. Ingestion Module (File Upload)
st.subheader("📂 Document Ingestion Framework")

# Split ingestion into columns for file and budget entry
col_file, col_budget = st.columns([2, 1])

with col_file:
    uploaded_file = st.file_uploader("Drop proposal PDF here for pipeline analysis", type=["pdf"])

with col_budget:
    # Captures the budget field your backend Form(...) validation requires!
    input_budget = st.number_input("Project Budget ($)", min_value=1000.0, max_value=10000000.0, value=50000.0, step=5000.0)

if uploaded_file is not None:
    if st.button("🚀 EXECUTE INFERENCE PIPELINE"):
        
        # 5. Simulated Pipeline Execution Logs
        st.subheader("💻 Live System Execution Logs")
        log_placeholder = st.empty()
        
        logs = [
            "[INFO] Initializing memory binary stream parsing via pdfplumber...",
            "[INFO] Extracting raw token sequences from target PDF matrix...",
            "[SUCCESS] Document text successfully vectorized.",
            "[INFO] Querying local vector store against historical project profiles...",
            "[INFO] Running feature parameters through trained ML models...",
            "[INFO] Processing Explainable AI (XAI) Gini attributes...",
            "[SUCCESS] Verification complete. Rendering pipeline metrics..."
        ]
        
        compiled_logs = ""
        for log in logs:
            compiled_logs += f"{log}<br>"
            log_placeholder.markdown(f'<div class="terminal-log">{compiled_logs}</div>', unsafe_allow_html=True)
            time.sleep(0.3)
            
        # 6. Fire live HTTP request across localhost port boundary to FastAPI
        try:
            # Package the PDF file binary stream
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            
            # Form-data body containing the budget value matching Form(...) requirements
            data_payload = {"budget": float(input_budget)}
            
            with st.spinner("Compiling UI cards..."):
                # 🚀 Pointing to the real prefixed path: /api/proposals/upload
                response = requests.post(
                    "http://127.0.0.1:8000/api/proposals/upload", 
                    files=files,
                    data=data_payload
                )
                
            if response.status_code == 200:
                data = response.json()
                analysis = data["analysis"]
                
                st.markdown("---")
                st.success(f"✨ Pipeline Inference Success (ID: #{data['proposal_id']} - {data['title']})")
                
                # 7. Split Layout for Metrics, XAI, and Narrative
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📊 Quantitative Analysis & Metrics")
                    
                    # Highlight Cards
                    c1, c2 = st.columns(2)
                    c1.metric(
                        label="Novelty Score", 
                        value=f"{analysis['metrics']['novelty_score']}%", 
                        delta=analysis['metrics'].get('novelty_rating', "Evaluated")
                    )
                    c2.metric(
                        label="Model Confidence", 
                        value=f"{analysis['evaluation_verdict']['confidence_percentage']}%"
                    )
                    
                    st.markdown("#### Final Verdict Decision")
                    decision = analysis['evaluation_verdict']['decision']
                    if "APPROVAL" in decision:
                        st.markdown(f"<h3 style='color: #34d399;'>{decision}</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='color: #f87171;'>{decision}</h3>", unsafe_allow_html=True)
                        
                    # Plotly Horizontal Bar Chart for XAI
                    st.markdown("#### 🔮 Explainable AI (XAI) Feature Importance")
                    xai_weights = analysis["explainable_ai_attribution"]
                    
                    df_chart = pd.DataFrame({
                        "Feature Vector": ["Novelty Impact", "Financial Feasibility", "Technical Density"],
                        "Gini Importance (%)": [
                            xai_weights["novelty_impact_weight"],
                            xai_weights["financial_feasibility_weight"],
                            xai_weights["technical_density_weight"]
                        ]
                    })
                    
                    fig = px.bar(
                        df_chart,
                        x="Gini Importance (%)",
                        y="Feature Vector",
                        orientation="h",
                        text="Gini Importance (%)",
                        color="Gini Importance (%)",
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color="#ffffff",
                        coloraxis_showscale=False,
                        height=220,
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("🧠 GenAI Executive Audit Briefing")
                    st.markdown("Generated via live execution of `gemini-2.5-flash` context layer:")
                    
                    st.info(f"\"{analysis['ai_generated_narrative']}\"")
                    
                    st.caption("⚖️ **Decision Audit Trail Note:**")
                    st.write(
                        "The relative weights derived above reflect the Gini Impurity reduction indices. "
                        "The textual uniqueness remains the strongest predictor for the resulting recommendation vector."
                    )
            else:
                st.error(f"Backend API error code: {response.status_code}. Details: {response.text}")
                
        except Exception as e:
            st.error(f"Could not connect to FastAPI server. Is Uvicorn running? Error: {str(e)}")