import streamlit as st
import pandas as pd
from logic import CESSystem

# ==========================================
# 1. SETUP (Runs once)
# ==========================================

st.set_page_config(page_title="CES System", layout="wide")

@st.cache_resource
def get_system():
    # This calls our class from logic.py
    return CESSystem()

# Load the system (and train models)
ces = get_system()

# ==========================================
# 2. UI LAYOUT
# ==========================================

st.title("🎓 CES: Contextual Embedding System")
st.markdown("### A Query Repair System for Ambiguous Inputs")
st.markdown("---")

# create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Context History")
    st.info("This simulates the previous messages in a chat.")
    # Input for context
    context_input = st.text_area(
        "Previous Conversation / User Mindset:", 
        value="I was just checking the exam schedule for this semester.",
        height=150
    )

with col2:
    st.subheader("2. Current User Query")
    # Input for query
    user_query = st.text_input("Enter a query (try something vague):", "When is it?")
    
    run_btn = st.button("🚀 Process Query", type="primary")

st.markdown("---")

# ==========================================
# 3. EXECUTION LOGIC
# ==========================================

if run_btn:
    st.header("3. System Results")
    
    # --- PHASE A: DETECTION ---
    is_missing = ces.detect_missing_context(user_query)
    
    if is_missing:
        st.error(f"⚠️ **Detection:** The system flagged '{user_query}' as AMBIGUOUS (Missing Context).")
        
        # --- PHASE B: GENERATION ---
        cfs = ces.generate_counterfactuals(user_query)
        st.write(f"**Generated {len(cfs)} possible interpretations (Counterfactuals)...**")
        
        # --- PHASE C: RANKING ---
        ranked_cfs = ces.rank_interpretations(user_query, cfs, context_input)
        
        # Get the best one
        best_match = ranked_cfs[0][0]
        best_score = ranked_cfs[0][1]
        
        # --- PHASE D: DISPLAY ---
        
        # 1. Show the Winner
        st.success(f"✅ **Repaired Query:** {best_match}")
        st.caption(f"Confidence Score: {best_score:.4f}")
        
        # 2. Show the detailed rankings table
        with st.expander("See Internal Ranking Logic (Click to Open)"):
            st.write("The system compared the Query + Context against all generated options.")
            df_results = pd.DataFrame(ranked_cfs, columns=["Interpretation", "Similarity Score"])
            st.dataframe(df_results)
            
    else:
        st.success(f"✅ **Detection:** The query '{user_query}' is Complete. No repair needed.")
        st.balloons()