import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Personality AI Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR HIGH-AESTHETIC MODERN UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Padding */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.25) 0%, rgba(59, 130, 246, 0.15) 100%);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A78BFA, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 400;
    }

    /* Card Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    
    .glass-card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Result Highlight Card */
    .result-card-introvert {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.3) 0%, rgba(79, 70, 229, 0.2) 100%);
        border: 2px solid #8B5CF6;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(139, 92, 246, 0.25);
    }
    
    .result-card-extrovert {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(14, 165, 233, 0.2) 100%);
        border: 2px solid #3B82F6;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.25);
    }

    .result-badge {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0.5rem 0;
    }
    
    .badge-introvert {
        color: #C4B5FD;
        text-shadow: 0 0 15px rgba(196, 181, 253, 0.5);
    }
    
    .badge-extrovert {
        color: #93C5FD;
        text-shadow: 0 0 15px rgba(147, 197, 253, 0.5);
    }
    
    .confidence-text {
        font-size: 1.1rem;
        color: #CBD5E1;
        font-weight: 500;
    }

    /* Trait Metric Pill */
    .metric-container {
        display: flex;
        justify-content: space-around;
        gap: 1rem;
        margin-top: 1rem;
    }

    .metric-pill {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        flex: 1;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    /* Hide Streamlit default menu/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model_filename = 'calibrated_model.joblib'
    if not os.path.exists(model_filename):
        # Fallback search in Hugging face folder if path is nested
        fallback_path = os.path.join('Hugging face', model_filename)
        if os.path.exists(fallback_path):
            model_filename = fallback_path
        else:
            return None
    try:
        return joblib.load(model_filename)
    except Exception as e:
        st.error(f"Error loading machine learning model: {e}")
        return None

model = load_model()

# --- HEADER SECTION ---
st.markdown("""
<div class="header-box">
    <div class="header-title">🧠 Personality Risk & Trait Predictor</div>
    <div class="header-subtitle">Interactive AI Assessment to estimate Introversion vs. Extroversion Likelihood</div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file `calibrated_model.joblib` was not found. Please ensure the model file is included in your repository root.")
    st.stop()

# --- APP LAYOUT: DUAL COLUMN ---
col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-title">📝 Daily Behavioral Habits</div>', unsafe_allow_html=True)
    
    alone_time = st.slider(
        "🧘 Hours Spent Alone Daily",
        min_value=0.0, max_value=24.0, value=6.0, step=0.5,
        help="Average hours you spend by yourself without active socializing."
    )
    
    events = st.slider(
        "🎉 Social Event Attendance Frequency (per month)",
        min_value=0, max_value=10, value=2, step=1,
        help="Number of social gatherings, parties, or meetups attended monthly."
    )
    
    going_out = st.slider(
        "🚶 Days Going Outside (per week)",
        min_value=0, max_value=7, value=3, step=1,
        help="Days per week you step outside for leisure, work, or activities."
    )

    posts = st.slider(
        "📱 Social Media Post Frequency (per week)",
        min_value=0, max_value=10, value=2, step=1,
        help="Average posts, stories, or public updates created weekly."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-title">🎭 Social Dynamics & Mindset</div>', unsafe_allow_html=True)
    
    col_sub1, col_sub2 = st.columns(2)
    with col_sub1:
        stage_fear = st.radio(
            "🎤 Do you experience Stage Fear?",
            options=["Yes", "No"],
            index=0,
            horizontal=True
        )
    with col_sub2:
        drained = st.radio(
            "🔋 Feel Drained After Socializing?",
            options=["Yes", "No"],
            index=0,
            horizontal=True
        )
        
    friends = st.number_input(
        "👥 Close Friends Circle Size",
        min_value=0, max_value=100, value=4, step=1,
        help="Number of close friends in your active circle."
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- PREDICTION LOGIC ---
user_data = pd.DataFrame([{
    'Time_spent_Alone': alone_time,
    'Stage_fear': 1 if stage_fear == 'Yes' else 0,
    'Social_event_attendance': events,
    'Going_outside': going_out,
    'Drained_after_socializing': 1 if drained == 'Yes' else 0,
    'Friends_circle_size': friends,
    'Post_frequency': posts
}])

try:
    probabilities = model.predict_proba(user_data)[0]
    p_introvert = float(probabilities[0])
    p_extrovert = float(probabilities[1])
    
    dominant = "Introvert" if p_introvert >= p_extrovert else "Extrovert"
    confidence = max(p_introvert, p_extrovert) * 100
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# --- RESULTS DISPLAY COLUMN ---
with col_result:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-title">✨ Personality Classification</div>', unsafe_allow_html=True)
    
    card_class = "result-card-introvert" if dominant == "Introvert" else "result-card-extrovert"
    badge_class = "badge-introvert" if dominant == "Introvert" else "badge-extrovert"
    icon = "🌌" if dominant == "Introvert" else "☀️"
    
    st.markdown(f"""
    <div class="{card_class}">
        <div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; color: #94A3B8;">Predicted Trait</div>
        <div class="result-badge {badge_class}">{icon} {dominant}</div>
        <div class="confidence-text">Model Confidence: <strong>{confidence:.1f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- PROBABILITY BREAKDOWN CHART ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-title">📊 Probability Distribution</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=['Extrovert', 'Introvert'],
        x=[p_extrovert * 100, p_introvert * 100],
        orientation='h',
        marker=dict(
            color=['#3B82F6', '#8B5CF6'],
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=[f"{p_extrovert * 100:.1f}%", f"{p_introvert * 100:.1f}%"],
        textposition='inside',
        insidetextanchor='end',
        textfont=dict(size=14, color='white', family='Inter')
    ))
    
    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=14, color='#F8FAFC', family='Inter')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=140
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- METRIC PILLS & INSIGHTS ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-title">💡 Behavioral Index</div>', unsafe_allow_html=True)
    
    solitude_index = round((alone_time / 24.0) * 100)
    social_index = round((going_out / 7.0 + events / 10.0) / 2.0 * 100)
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-pill">
            <div class="metric-label">Solitude Index</div>
            <div class="metric-value">{solitude_index}%</div>
        </div>
        <div class="metric-pill">
            <div class="metric-label">Outing Energy</div>
            <div class="metric-value">{social_index}%</div>
        </div>
        <div class="metric-pill">
            <div class="metric-label">Social Battery</div>
            <div class="metric-value">{"Low" if drained == "Yes" else "High"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Insights summary
    if dominant == "Introvert":
        summary = "You thrive best in tranquil environments with select close connections. You gain energy through focused personal time rather than large crowds."
    else:
        summary = "You derive energy from active social interactions, group dynamics, and frequent external activities. You feel most refreshed around people."
        
    st.info(f"**Insight:** {summary}")
    st.markdown('</div>', unsafe_allow_html=True)
