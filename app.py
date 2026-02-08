"""
Fuzzy Expert System - Student Performance Evaluation
=====================================================

Interactive step-by-step wizard demonstrating fuzzy inference.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from fuzzy_expert_system import FuzzyStudentEvaluator

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Fuzzy Expert System - Student Evaluation",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 0.5rem 0;
    }
    
    .step-badge {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 2rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .result-high {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .result-medium {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .result-low {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .rule-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-left: 4px solid #6366f1;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        font-family: monospace;
    }
    
    .rule-high { border-left-color: #22c55e; }
    .rule-medium { border-left-color: #f59e0b; }
    .rule-low { border-left-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUZZY RULES DEFINITION (for display)
# =============================================================================

FUZZY_RULES = [
    {"id": "R1", "condition": "GPA is High AND Test Score is Excellent AND Participation is Active AND Absences are Few", "result": "HIGH", "explanation": "Top performers with excellent grades, test scores, active participation, and minimal absences achieve the highest evaluation."},
    {"id": "R2", "condition": "GPA is High AND Test Score is Good", "result": "HIGH", "explanation": "Students with high GPA and good test scores demonstrate strong academic performance."},
    {"id": "R3", "condition": "GPA is High AND Test Score is Average", "result": "MEDIUM", "explanation": "High GPA with average test scores suggests inconsistent performance, resulting in medium evaluation."},
    {"id": "R4", "condition": "GPA is Medium AND Test Score is Average AND Participation is Moderate", "result": "MEDIUM", "explanation": "Average performance across multiple dimensions leads to medium evaluation."},
    {"id": "R5", "condition": "GPA is Medium AND Participation is Active AND Absences are Few", "result": "MEDIUM", "explanation": "Active engagement and attendance can compensate for medium GPA."},
    {"id": "R6", "condition": "GPA is Low AND Test Score is Poor", "result": "LOW", "explanation": "Low GPA combined with poor test scores indicates struggling performance."},
    {"id": "R7", "condition": "Absences are Many AND Participation is Passive", "result": "LOW", "explanation": "High absenteeism and lack of participation strongly indicate poor performance."},
    {"id": "R8", "condition": "GPA is Low AND Participation is Active AND Test Score is Average", "result": "MEDIUM", "explanation": "Active participation can partially compensate for low GPA."},
    {"id": "R9", "condition": "GPA is Medium AND Test Score is Excellent", "result": "HIGH", "explanation": "Excellent test performance can elevate medium GPA students."},
    {"id": "R10", "condition": "Test Score is Good AND Absences are Few AND Participation is Moderate", "result": "MEDIUM", "explanation": "Good test scores with attendance but moderate participation yields medium evaluation."},
    {"id": "R11", "condition": "Test Score is Poor AND Participation is Passive", "result": "LOW", "explanation": "Poor test scores combined with lack of participation indicates at-risk students."},
    {"id": "R12", "condition": "GPA is Medium AND Test Score is Good AND Absences are Moderate", "result": "MEDIUM", "explanation": "Good test scores offset moderate absences for medium GPA students."},
]

# =============================================================================
# FUZZY SYSTEM
# =============================================================================

@st.cache_resource
def create_fuzzy_system():
    """
    Create the fuzzy inference system using the shared FuzzyStudentEvaluator class.
    This ensures consistency between the Streamlit app and the core module.
    """
    return FuzzyStudentEvaluator()

def evaluate_student(fuzzy_sys, gpa, absences, participation, test_score):
    try:
        sim = ctrl.ControlSystemSimulation(fuzzy_sys['system'])
        sim.input['gpa'] = np.clip(gpa, 2, 6)
        sim.input['absences'] = np.clip(absences, 0, 30)
        sim.input['participation'] = np.clip(participation, 0, 100)
        sim.input['test_score'] = np.clip(test_score, 0, 100)
        sim.compute()
        return sim.output['performance']
    except:
        return 50.0

def get_category(score):
    if score >= 70: return "HIGH", "#10b981"
    elif score >= 40: return "MEDIUM", "#f59e0b"
    else: return "LOW", "#ef4444"

def get_membership_degrees(fuzzy_sys, gpa_val, abs_val, part_val, test_val):
    return {
        'gpa': {
            'low': float(fuzz.interp_membership(fuzzy_sys['gpa'].universe, fuzzy_sys['gpa']['low'].mf, gpa_val)),
            'medium': float(fuzz.interp_membership(fuzzy_sys['gpa'].universe, fuzzy_sys['gpa']['medium'].mf, gpa_val)),
            'high': float(fuzz.interp_membership(fuzzy_sys['gpa'].universe, fuzzy_sys['gpa']['high'].mf, gpa_val)),
        },
        'absences': {
            'few': float(fuzz.interp_membership(fuzzy_sys['absences'].universe, fuzzy_sys['absences']['few'].mf, abs_val)),
            'moderate': float(fuzz.interp_membership(fuzzy_sys['absences'].universe, fuzzy_sys['absences']['moderate'].mf, abs_val)),
            'many': float(fuzz.interp_membership(fuzzy_sys['absences'].universe, fuzzy_sys['absences']['many'].mf, abs_val)),
        },
        'participation': {
            'passive': float(fuzz.interp_membership(fuzzy_sys['participation'].universe, fuzzy_sys['participation']['passive'].mf, part_val)),
            'moderate': float(fuzz.interp_membership(fuzzy_sys['participation'].universe, fuzzy_sys['participation']['moderate'].mf, part_val)),
            'active': float(fuzz.interp_membership(fuzzy_sys['participation'].universe, fuzzy_sys['participation']['active'].mf, part_val)),
        },
        'test_score': {
            'poor': float(fuzz.interp_membership(fuzzy_sys['test_score'].universe, fuzzy_sys['test_score']['poor'].mf, test_val)),
            'average': float(fuzz.interp_membership(fuzzy_sys['test_score'].universe, fuzzy_sys['test_score']['average'].mf, test_val)),
            'good': float(fuzz.interp_membership(fuzzy_sys['test_score'].universe, fuzzy_sys['test_score']['good'].mf, test_val)),
            'excellent': float(fuzz.interp_membership(fuzzy_sys['test_score'].universe, fuzzy_sys['test_score']['excellent'].mf, test_val)),
        }
    }

def plot_membership(fuzzy_sys, var_name, input_val, title):
    var = fuzzy_sys[var_name]
    fig = go.Figure()
    colors = {'low': '#ef4444', 'medium': '#f59e0b', 'high': '#22c55e',
              'few': '#22c55e', 'moderate': '#f59e0b', 'many': '#ef4444',
              'passive': '#ef4444', 'active': '#22c55e',
              'poor': '#ef4444', 'average': '#f59e0b', 'good': '#22c55e', 'excellent': '#3b82f6'}
    
    for term in var.terms:
        mf = var[term].mf
        color = colors.get(term, '#6366f1')
        fig.add_trace(go.Scatter(x=var.universe, y=mf, name=term.capitalize(),
                                  line=dict(color=color, width=2), fill='tozeroy',
                                  fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}"))
    
    fig.add_vline(x=input_val, line_dash="dash", line_color="white", line_width=3,
                  annotation_text=f"Input: {input_val}", annotation_position="top")
    fig.update_layout(title=title, template="plotly_dark", height=280,
                      margin=dict(l=40, r=40, t=50, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      xaxis_title=var_name.replace('_', ' ').title(), yaxis_title="μ(x)")
    return fig

def plot_membership_only(fuzzy_sys, var_name, title):
    """Plot membership function without input value marker."""
    var = fuzzy_sys[var_name]
    fig = go.Figure()
    colors = {'low': '#ef4444', 'medium': '#f59e0b', 'high': '#22c55e',
              'few': '#22c55e', 'moderate': '#f59e0b', 'many': '#ef4444',
              'passive': '#ef4444', 'active': '#22c55e',
              'poor': '#ef4444', 'average': '#f59e0b', 'good': '#22c55e', 'excellent': '#3b82f6'}
    
    for term in var.terms:
        mf = var[term].mf
        color = colors.get(term, '#6366f1')
        fig.add_trace(go.Scatter(x=var.universe, y=mf, name=term.capitalize(),
                                  line=dict(color=color, width=2), fill='tozeroy',
                                  fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}"))
    
    fig.update_layout(title=title, template="plotly_dark", height=250,
                      margin=dict(l=40, r=40, t=50, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      xaxis_title=var_name.replace('_', ' ').title(), yaxis_title="μ(x)")
    return fig

# =============================================================================
# ABOUT TAB CONTENT
# =============================================================================

def render_about_tab(fuzzy_sys):
    """Render the About tab with system explanation and rules."""
    
    st.header("📖 About This Application")
    
    st.success("""
    **🎯 Purpose**
    
    This is a **Fuzzy Expert System** designed to evaluate student academic performance using 
    fuzzy logic and the **Compositional Rule of Inference** (Mamdani method).
    
    Instead of rigid pass/fail thresholds, fuzzy logic allows for **nuanced evaluation** 
    that mirrors how human experts would assess student performance.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔢 Input Variables")
        st.markdown("""
        The system considers **4 input factors**:
        
        | Variable | Range | Linguistic Terms |
        |----------|-------|------------------|
        | **GPA** | 2-6 (Bulgarian scale) | Low, Medium, High |
        | **Absences** | 0-30 classes | Few, Moderate, Many |
        | **Participation** | 0-100% | Passive, Moderate, Active |
        | **Test Score** | 0-100% | Poor, Average, Good, Excellent |
        """)
    
    with col2:
        st.subheader("📊 Output Variable")
        st.markdown("""
        The system produces a **Performance Score** (0-100):
        
        | Score Range | Category | Description |
        |-------------|----------|-------------|
        | 70-100 | 🟢 **HIGH** | Excellent performance |
        | 40-69 | 🟡 **MEDIUM** | Satisfactory |
        | 0-39 | 🔴 **LOW** | Needs improvement |
        """)
    
    st.divider()
    
    # Membership Functions
    st.subheader("📈 Membership Functions")
    st.markdown("These define how crisp values map to linguistic terms:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_membership_only(fuzzy_sys, 'gpa', 'GPA (2-6)'), use_container_width=True)
        st.plotly_chart(plot_membership_only(fuzzy_sys, 'participation', 'Participation (0-100%)'), use_container_width=True)
    with col2:
        st.plotly_chart(plot_membership_only(fuzzy_sys, 'absences', 'Absences (0-30)'), use_container_width=True)
        st.plotly_chart(plot_membership_only(fuzzy_sys, 'test_score', 'Test Score (0-100%)'), use_container_width=True)
    
    st.divider()
    
    # Fuzzy Rules
    st.subheader("📜 Fuzzy Rule Base")
    st.markdown("""
    The expert system uses **12 IF-THEN rules** that encode expert knowledge about student evaluation.
    These rules are applied using the **Mamdani inference method**:
    """)
    
    # Display rules in a nice format
    for rule in FUZZY_RULES:
        result_class = f"rule-{rule['result'].lower()}"
        st.markdown(f"""
        <div class="rule-card {result_class}">
            <strong>{rule['id']}:</strong> IF {rule['condition']} → THEN Performance is <strong>{rule['result']}</strong>
        </div>
        """, unsafe_allow_html=True)
        with st.expander(f"💡 Why this rule?"):
            st.write(rule['explanation'])
    
    st.divider()
    
    # Inference Method
    st.subheader("⚙️ Inference Method")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Mamdani Fuzzy Inference System:**
        
        1. **Fuzzification** - Convert crisp inputs to fuzzy membership degrees
        2. **Rule Evaluation** - Apply rules using MIN for AND, MAX for OR  
        3. **Aggregation** - Combine all rule outputs using MAX
        4. **Defuzzification** - Convert fuzzy output to crisp value using Centroid
        """)
    
    with col2:
        st.markdown("""
        **Key Formulas:**
        """)
        st.latex(r"\mu_{A \land B}(x) = \min(\mu_A(x), \mu_B(x))")
        st.latex(r"\mu_{output} = \max(\mu_{rule_1}, \mu_{rule_2}, ...)")
        st.latex(r"z^* = \frac{\int z \cdot \mu(z) dz}{\int \mu(z) dz}")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    fuzzy_sys = create_fuzzy_system()
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'selected_student_idx' not in st.session_state:
        st.session_state.selected_student_idx = 0
    
    # Header
    st.markdown('<h1 class="main-title">🎓 Fuzzy Expert System for Student Evaluation</h1>', unsafe_allow_html=True)
    st.caption("Interactive Step-by-Step Demonstration of Compositional Rule of Inference")
    
    # Main tabs
    main_tab1, main_tab2 = st.tabs(["🔬 Evaluation Wizard", "📖 About the System"])
    
    # ==========================================================================
    # ABOUT TAB
    # ==========================================================================
    with main_tab2:
        render_about_tab(fuzzy_sys)
    
    # ==========================================================================
    # EVALUATION WIZARD TAB
    # ==========================================================================
    with main_tab1:
        # Step indicator
        total_steps = 7
        step_names = ["Load Data", "Review Rules", "Process Data", "Select Student", "Fuzzification", "Rule Evaluation", "Final Result"]
        
        # Progress bar
        st.progress((st.session_state.step + 1) / total_steps)
        st.markdown(f'<div class="step-badge">Step {st.session_state.step + 1} of {total_steps}: {step_names[st.session_state.step]}</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # =====================================================================
        # STEP 0: LOAD DATA
        # =====================================================================
        if st.session_state.step == 0:
            st.header("📂 Step 1: Load Student Data")
            
            st.info("""
            **🎯 What is this step about?**
            
            Before we can evaluate student performance using fuzzy logic, we need **input data**. 
            This data contains measurable attributes for each student that our expert system will analyze.
            """)
            
            with st.expander("📊 Required Data Format", expanded=True):
                st.markdown("""
                The CSV file should contain the following columns:
                
                | Column | Range | Description |
                |--------|-------|-------------|
                | **GPA** | 2-6 | Grade Point Average (Bulgarian scale: 2=fail, 6=excellent) |
                | **Absences** | 0-30 | Number of classes missed |
                | **Participation** | 0-100 | Class participation percentage |
                | **Test_Score** | 0-100 | Test/exam score percentage |
                """)
            
            st.subheader("Choose Data Source")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📤 Upload Your Data")
                uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
                
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(df)} students from uploaded file!")
            
            with col2:
                st.markdown("#### 📁 Or Use Sample Data")
                if st.button("🎲 Load Sample Data (120 students)", type="primary", use_container_width=True):
                    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_students.csv')
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                    else:
                        np.random.seed(42)
                        n = 120
                        df = pd.DataFrame({
                            'Student_ID': [f'STU-{i:03d}' for i in range(1, n+1)],
                            'Student_Name': [f'Student {i}' for i in range(1, n+1)],
                            'GPA': np.round(np.random.uniform(2.5, 6, n), 2),
                            'Absences': np.random.randint(0, 25, n),
                            'Participation': np.random.randint(20, 100, n),
                            'Test_Score': np.random.randint(30, 100, n)
                        })
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(df)} students from sample data!")
            
            if st.session_state.data_loaded:
                st.subheader("📋 Data Preview")
                st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        # =====================================================================
        # STEP 1: REVIEW RULES (NEW STEP)
        # =====================================================================
        elif st.session_state.step == 1:
            st.header("📜 Step 2: Review Fuzzy Rules")
            
            st.info("""
            **🎯 What is this step about?**
            
            Before processing the data, let's review the **expert rules** that will be applied. 
            These rules encode domain knowledge about how student attributes relate to academic performance.
            """)
            
            st.warning("""
            ⚠️ **Important:** These rules will be applied to ALL students in your dataset. 
            Each student's inputs will be evaluated against these 12 rules using fuzzy logic.
            """)
            
            # Summary of rules by outcome
            col1, col2, col3 = st.columns(3)
            with col1:
                high_rules = [r for r in FUZZY_RULES if r['result'] == 'HIGH']
                st.metric("🟢 Rules for HIGH", len(high_rules))
            with col2:
                medium_rules = [r for r in FUZZY_RULES if r['result'] == 'MEDIUM']
                st.metric("🟡 Rules for MEDIUM", len(medium_rules))
            with col3:
                low_rules = [r for r in FUZZY_RULES if r['result'] == 'LOW']
                st.metric("🔴 Rules for LOW", len(low_rules))
            
            st.subheader("📋 Complete Rule Base")
            
            # Create DataFrame for rules
            rules_df = pd.DataFrame([
                {"Rule": r['id'], "IF Condition": r['condition'], "THEN Result": r['result']}
                for r in FUZZY_RULES
            ])
            
            st.dataframe(rules_df, use_container_width=True, hide_index=True)
            
            st.subheader("🔍 Rule Details")
            
            for rule in FUZZY_RULES:
                result_emoji = "🟢" if rule['result'] == 'HIGH' else "🟡" if rule['result'] == 'MEDIUM' else "🔴"
                with st.expander(f"{result_emoji} **{rule['id']}**: IF {rule['condition'][:50]}... → {rule['result']}"):
                    st.markdown(f"**Full Condition:** {rule['condition']}")
                    st.markdown(f"**Result:** Performance is **{rule['result']}**")
                    st.markdown(f"**Explanation:** {rule['explanation']}")
        
        # =====================================================================
        # STEP 2: PROCESS DATA (was Step 1)
        # =====================================================================
        elif st.session_state.step == 2:
            st.header("📈 Step 3: Process Data & View Results")
            
            st.info("""
            **🎯 What is happening?**
            
            The fuzzy expert system processes **all students** using the rules you just reviewed.
            Each student's attributes are fuzzified, rules are evaluated, and a final score is computed.
            """)
            
            # Process all students
            df = st.session_state.df
            results = []
            
            progress = st.progress(0, text="Processing students...")
            for idx, row in df.iterrows():
                gpa = row.get('GPA', row.get('gpa', 3.0))
                absences = row.get('Absences', row.get('absences', 10))
                participation = row.get('Participation', row.get('participation', 50))
                test_score = row.get('Test_Score', row.get('test_score', row.get('Test Score', 60)))
                
                score = evaluate_student(fuzzy_sys, gpa, absences, participation, test_score)
                cat, _ = get_category(score)
                
                results.append({
                    'ID': row.get('Student_ID', row.get('student_id', f'STU-{idx+1:03d}')),
                    'Name': row.get('Student_Name', row.get('name', f'Student {idx+1}')),
                    'GPA': gpa, 'Absences': absences, 'Participation': participation,
                    'Test_Score': test_score, 'Score': round(score, 1), 'Category': cat
                })
                progress.progress((idx + 1) / len(df), text=f"Processing student {idx + 1}/{len(df)}...")
            
            progress.empty()
            st.session_state.results_df = pd.DataFrame(results)
            results_df = st.session_state.results_df
            
            # Summary metrics
            high_count = len(results_df[results_df['Category'] == 'HIGH'])
            medium_count = len(results_df[results_df['Category'] == 'MEDIUM'])
            low_count = len(results_df[results_df['Category'] == 'LOW'])
            
            st.subheader("📊 Summary Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Students", len(results_df))
            col2.metric("🟢 High", high_count, f"{100*high_count/len(results_df):.1f}%")
            col3.metric("🟡 Medium", medium_count, f"{100*medium_count/len(results_df):.1f}%")
            col4.metric("🔴 Low", low_count, f"{100*low_count/len(results_df):.1f}%")
            col5.metric("Average Score", f"{results_df['Score'].mean():.1f}")
            
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['High', 'Medium', 'Low'], values=[high_count, medium_count, low_count],
                    marker_colors=['#10b981', '#f59e0b', '#ef4444'], hole=0.4
                )])
                fig_pie.update_layout(title="Performance Distribution", template="plotly_dark", height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_hist = px.histogram(results_df, x='Score', nbins=20, color_discrete_sequence=['#6366f1'])
                fig_hist.update_layout(title="Score Distribution", template="plotly_dark", height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            st.subheader("📋 All Results")
            st.dataframe(results_df, use_container_width=True, height=250)
        
        # =====================================================================
        # STEP 3: SELECT STUDENT (was Step 2)
        # =====================================================================
        elif st.session_state.step == 3:
            st.header("👤 Step 4: Select Student for Detailed Analysis")
            
            st.info("""
            **🎯 What is this step about?**
            
            Now we'll examine **how the fuzzy inference works in detail** for a specific student.
            Select any student to see the complete step-by-step reasoning process.
            """)
            
            results_df = st.session_state.results_df
            
            # Student selector
            student_options = [f"{row['ID']} - {row['Name']} ({row['Category']})" for _, row in results_df.iterrows()]
            selected = st.selectbox("🔎 Select a student:", student_options, index=st.session_state.selected_student_idx)
            st.session_state.selected_student_idx = student_options.index(selected)
            
            student = results_df.iloc[st.session_state.selected_student_idx]
            st.session_state.student = student
            
            # Show selected student
            st.subheader(f"👤 {student['Name']}")
            st.caption(f"ID: {student['ID']}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📚 GPA", f"{student['GPA']:.2f}")
            col2.metric("🚫 Absences", int(student['Absences']))
            col3.metric("🙋 Participation", f"{student['Participation']}%")
            col4.metric("📝 Test Score", f"{student['Test_Score']}%")
        
        # =====================================================================
        # STEP 4: FUZZIFICATION (was Step 3)
        # =====================================================================
        elif st.session_state.step == 4:
            student = st.session_state.student
            
            st.header("🔄 Step 5: Fuzzification")
            st.caption(f"Converting crisp inputs to fuzzy memberships for {student['Name']}")
            
            st.info("""
            **🎯 What is Fuzzification?**
            
            Fuzzification transforms precise numerical values into **degrees of membership** in fuzzy sets.
            This allows the rules to handle imprecise, human-like reasoning.
            """)
            
            memberships = get_membership_degrees(fuzzy_sys, student['GPA'], student['Absences'],
                                                 student['Participation'], student['Test_Score'])
            st.session_state.memberships = memberships
            
            # Plots
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_membership(fuzzy_sys, 'gpa', student['GPA'], f"GPA = {student['GPA']:.2f}")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Membership Degrees:**")
                for term, val in memberships['gpa'].items():
                    bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                    st.code(f"{term.upper():8} {bar} {val:.3f}")
            
            with col2:
                fig = plot_membership(fuzzy_sys, 'absences', student['Absences'], f"Absences = {int(student['Absences'])}")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Membership Degrees:**")
                for term, val in memberships['absences'].items():
                    bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                    st.code(f"{term.upper():8} {bar} {val:.3f}")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_membership(fuzzy_sys, 'participation', student['Participation'], f"Participation = {student['Participation']}%")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Membership Degrees:**")
                for term, val in memberships['participation'].items():
                    bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                    st.code(f"{term.upper():8} {bar} {val:.3f}")
            
            with col2:
                fig = plot_membership(fuzzy_sys, 'test_score', student['Test_Score'], f"Test Score = {student['Test_Score']}%")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Membership Degrees:**")
                for term, val in memberships['test_score'].items():
                    bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                    st.code(f"{term.upper():9} {bar} {val:.3f}")
        
        # =====================================================================
        # STEP 5: RULE EVALUATION (was Step 4)
        # =====================================================================
        elif st.session_state.step == 5:
            student = st.session_state.student
            memberships = st.session_state.memberships
            
            st.header("⚡ Step 6: Rule Evaluation")
            st.caption(f"Applying IF-THEN rules for {student['Name']}")
            
            st.info("""
            **🎯 What is Rule Evaluation?**
            
            Each rule's conditions are evaluated using the membership degrees from fuzzification.
            The AND operation uses MIN, the OR operation uses MAX.
            """)
            
            # Calculate rule activations
            rules_info = [
                ("R1", "GPA=High ∧ Test=Excellent ∧ Part=Active ∧ Abs=Few", "HIGH",
                 min(memberships['gpa']['high'], memberships['test_score']['excellent'], 
                     memberships['participation']['active'], memberships['absences']['few'])),
                ("R2", "GPA=High ∧ Test=Good", "HIGH",
                 min(memberships['gpa']['high'], memberships['test_score']['good'])),
                ("R3", "GPA=High ∧ Test=Average", "MEDIUM",
                 min(memberships['gpa']['high'], memberships['test_score']['average'])),
                ("R4", "GPA=Medium ∧ Test=Average ∧ Part=Moderate", "MEDIUM",
                 min(memberships['gpa']['medium'], memberships['test_score']['average'], memberships['participation']['moderate'])),
                ("R5", "GPA=Medium ∧ Part=Active ∧ Abs=Few", "MEDIUM",
                 min(memberships['gpa']['medium'], memberships['participation']['active'], memberships['absences']['few'])),
                ("R6", "GPA=Low ∧ Test=Poor", "LOW",
                 min(memberships['gpa']['low'], memberships['test_score']['poor'])),
                ("R7", "Abs=Many ∧ Part=Passive", "LOW",
                 min(memberships['absences']['many'], memberships['participation']['passive'])),
                ("R8", "GPA=Low ∧ Part=Active ∧ Test=Average", "MEDIUM",
                 min(memberships['gpa']['low'], memberships['participation']['active'], memberships['test_score']['average'])),
                ("R9", "GPA=Medium ∧ Test=Excellent", "HIGH",
                 min(memberships['gpa']['medium'], memberships['test_score']['excellent'])),
                ("R10", "Test=Good ∧ Abs=Few ∧ Part=Moderate", "MEDIUM",
                 min(memberships['test_score']['good'], memberships['absences']['few'], memberships['participation']['moderate'])),
                ("R11", "Test=Poor ∧ Part=Passive", "LOW",
                 min(memberships['test_score']['poor'], memberships['participation']['passive'])),
                ("R12", "GPA=Medium ∧ Test=Good ∧ Abs=Moderate", "MEDIUM",
                 min(memberships['gpa']['medium'], memberships['test_score']['good'], memberships['absences']['moderate'])),
            ]
            st.session_state.rules_info = rules_info
            
            st.subheader("📋 Rule Activations")
            
            rules_df = pd.DataFrame([{'Rule': r[0], 'IF': r[1], 'THEN': r[2], 'Activation': r[3]} for r in rules_info])
            rules_df = rules_df.sort_values('Activation', ascending=False)
            
            st.dataframe(rules_df.style.background_gradient(subset=['Activation'], cmap='YlOrRd'),
                         use_container_width=True, hide_index=True)
            
            # Active rules chart
            active = [(r[0], r[3], r[2]) for r in rules_info if r[3] > 0.001]
            if active:
                st.subheader("📊 Active Rules (Activation > 0)")
                fig = go.Figure()
                colors = {'HIGH': '#22c55e', 'MEDIUM': '#f59e0b', 'LOW': '#ef4444'}
                for rule in sorted(active, key=lambda x: x[1], reverse=True):
                    fig.add_trace(go.Bar(x=[rule[1]], y=[rule[0]], orientation='h',
                                          marker_color=colors[rule[2]], name=f"{rule[0]}→{rule[2]}",
                                          text=f"{rule[1]:.3f}", textposition='outside'))
                fig.update_layout(template="plotly_dark", height=max(250, len(active)*45), 
                                  showlegend=False, xaxis_title="Activation Strength")
                st.plotly_chart(fig, use_container_width=True)
        
        # =====================================================================
        # STEP 6: FINAL RESULT (was Step 6)
        # =====================================================================
        elif st.session_state.step == 6:
            student = st.session_state.student
            
            st.header("🎯 Step 7: Final Result")
            
            st.success(f"""
            **🏁 Complete Inference for {student['Name']}**
            
            All rule outputs have been aggregated using MAX operation, then defuzzified using the Centroid method.
            """)
            
            # Result display
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=student['Score'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Performance Score", 'font': {'size': 24, 'color': 'white'}},
                    number={'font': {'size': 60, 'color': 'white'}},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#6366f1'},
                           'steps': [{'range': [0, 40], 'color': '#ef4444'},
                                     {'range': [40, 70], 'color': '#f59e0b'},
                                     {'range': [70, 100], 'color': '#22c55e'}]}
                ))
                fig.update_layout(template="plotly_dark", height=350, margin=dict(t=80, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                cat = student['Category']
                if cat == "HIGH":
                    st.markdown('<div class="result-high">🏆 HIGH PERFORMANCE</div>', unsafe_allow_html=True)
                elif cat == "MEDIUM":
                    st.markdown('<div class="result-medium">📊 MEDIUM PERFORMANCE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-low">⚠️ LOW PERFORMANCE</div>', unsafe_allow_html=True)
            
            st.divider()
            
            st.subheader("📋 Student Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("GPA", f"{student['GPA']:.2f}")
            col2.metric("Absences", int(student['Absences']))
            col3.metric("Participation", f"{student['Participation']}%")
            col4.metric("Test Score", f"{student['Test_Score']}%")
            col5.metric("Final Score", f"{student['Score']:.1f}")
            
            st.info("""
            **🔄 What's Next?**
            
            - Click **"Start Over"** to analyze a new dataset
            - Click **"⬅️ Back"** to review any step
            - Go to **"📖 About the System"** tab to review the rules and methodology
            """)
        
        # =====================================================================
        # NAVIGATION BUTTONS
        # =====================================================================
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state.step > 0:
                if st.button("⬅️ Back", use_container_width=True):
                    st.session_state.step -= 1
                    st.rerun()
        
        with col3:
            if st.session_state.step == 0:
                if st.session_state.data_loaded:
                    if st.button("Next ➡️", type="primary", use_container_width=True):
                        st.session_state.step += 1
                        st.rerun()
                else:
                    st.button("Next ➡️", type="primary", use_container_width=True, disabled=True)
            elif st.session_state.step < 6:
                if st.button("Next ➡️", type="primary", use_container_width=True):
                    st.session_state.step += 1
                    st.rerun()
            else:
                if st.button("🔄 Start Over", type="primary", use_container_width=True):
                    st.session_state.step = 0
                    st.session_state.data_loaded = False
                    st.rerun()

if __name__ == "__main__":
    main()
