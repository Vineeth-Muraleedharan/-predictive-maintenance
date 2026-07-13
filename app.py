"""
app.py - Predictive Maintenance - Failure Prediction App

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import pickle
import base64
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title='Predictive Maintenance',
    page_icon='⚙️',
    layout='centered'
)

# ── Background image ──────────────────────────────────────────
def add_bg_image(image_file):
    try:
        with open(image_file, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url('data:image/png;base64,{encoded}');
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass  # skip if image not found

add_bg_image('Predictive-Maintenance-AI.png')

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .title {
        font-size: 2rem; font-weight: 700;
        color: #ffffff; text-align: center;
        text-shadow: 0 2px 8px rgba(0,0,0,0.8);
    }
    .subtitle {
        font-size: 0.95rem; color: #e0e0e0;
        text-align: center; margin-bottom: 2rem;
        text-shadow: 0 1px 4px rgba(0,0,0,0.8);
    }
    .failure-box {
        background: rgba(253,237,236,0.95);
        border-left: 5px solid #E74C3C;
        border-radius: 8px; padding: 1rem;
        font-size: 1.1rem; font-weight: 600; color: #C0392B;
    }
    .safe-box {
        background: rgba(234,250,241,0.95);
        border-left: 5px solid #27AE60;
        border-radius: 8px; padding: 1rem;
        font-size: 1.1rem; font-weight: 600; color: #1E8449;
    }
    .warning-box {
        background: rgba(254,249,231,0.95);
        border-left: 5px solid #F39C12;
        border-radius: 8px; padding: 1rem;
        font-size: 1.1rem; font-weight: 600; color: #D68910;
    }
    section[data-testid="stSidebar"] {
        background: rgba(13,17,23,0.85);
    }
    .block-container {
        background: rgba(13,17,23,0.75);
        border-radius: 12px;
        padding: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load saved model ──────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('best_maintenance_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_maintenance.pkl', 'rb') as f:
        scaler = pickle.load(f)
    feature_names = [
        'Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'Temp_Diff',
        'Power',
        'Wear_Torque'
    ]
    return model, scaler, feature_names


model, scaler, feature_names = load_model()


# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="title">⚙️ Machine Failure Predictor</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle"> Predictive Maintenance</div>',
    unsafe_allow_html=True)
st.markdown('---')


# ── Input ─────────────────────────────────────────────────────
st.subheader('Enter Sensor Readings')

c1, c2 = st.columns(2)

with c1:
    machine_type = st.selectbox(
        'Machine Type',
        ['L - Light Duty', 'M - Medium Duty', 'H - Heavy Duty']
    )
    type_val = {
        'L - Light Duty':  0,
        'M - Medium Duty': 1,
        'H - Heavy Duty':  2
    }[machine_type]

    air_temp = st.number_input(
        'Air Temperature (K)',
        min_value=295.0, max_value=305.0,
        value=298.0, step=0.1
    )

    proc_temp = st.number_input(
        'Process Temperature (K)',
        min_value=305.0, max_value=315.0,
        value=308.0, step=0.1
    )

with c2:
    rot_speed = st.number_input(
        'Rotational Speed (rpm)',
        min_value=1168, max_value=2886,
        value=1550, step=10
    )

    torque = st.number_input(
        'Torque (Nm)',
        min_value=3.8, max_value=76.6,
        value=45.0, step=0.1
    )

    tool_wear = st.number_input(
        'Tool Wear (min)',
        min_value=0, max_value=253,
        value=150, step=1
    )

st.markdown('---')
predict_btn = st.button('🔍 Predict Failure Risk', use_container_width=True)


# ── Prediction ────────────────────────────────────────────────
if predict_btn:
    # Compute engineered features
    temp_diff   = proc_temp - air_temp
    power       = torque * rot_speed
    wear_torque = tool_wear * torque

    input_dict = {
        'Type':                    type_val,
        'Air temperature [K]':     air_temp,
        'Process temperature [K]': proc_temp,
        'Rotational speed [rpm]':  rot_speed,
        'Torque [Nm]':             torque,
        'Tool wear [min]':         tool_wear,
        'Temp_Diff':               temp_diff,
        'Power':                   power,
        'Wear_Torque':             wear_torque
    }

    # Predict — 0.3 threshold for early warning
    df_input = pd.DataFrame([input_dict])[feature_names]
    scaled   = scaler.transform(df_input)
    prob     = model.predict_proba(scaled)[0][1]

    # Result
    st.subheader('Prediction Result')

    if prob >= 0.5:
        st.markdown(
            f'<div class="failure-box">⚠️ HIGH FAILURE RISK DETECTED<br>'
            f'Failure Probability: {prob*100:.1f}%</div>',
            unsafe_allow_html=True)
        st.error('Immediate maintenance inspection recommended!')

    elif prob >= 0.3:
        st.markdown(
            f'<div class="warning-box">🔶 EARLY WARNING - MONITOR CLOSELY<br>'
            f'Failure Probability: {prob*100:.1f}%</div>',
            unsafe_allow_html=True)
        st.warning('Schedule maintenance check soon.')

    else:
        st.markdown(
            f'<div class="safe-box">✅ MACHINE OPERATING NORMALLY<br>'
            f'Failure Probability: {prob*100:.1f}%</div>',
            unsafe_allow_html=True)
        st.success('No immediate action required.')

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=round(prob * 100, 1),
        title={'text': 'Failure Probability (%)'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar':  {'color': '#E74C3C' if prob >= 0.5
                              else '#F39C12' if prob >= 0.3
                              else '#27AE60'},
            'steps': [
                {'range': [0,  30],  'color': '#EAFAF1'},
                {'range': [30, 50],  'color': '#FEF9E7'},
                {'range': [50, 100], 'color': '#FDEDEC'}
            ],
            'threshold': {
                'line':      {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value':     50
            }
        }
    ))
    fig.update_layout(
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Computed features
    with st.expander('View auto-computed engineered features'):
        st.dataframe(pd.DataFrame({
            'Feature': ['Temp_Diff', 'Power', 'Wear_Torque'],
            'Value':   [
                f'{temp_diff:.2f} K',
                f'{power:,.0f}',
                f'{wear_torque:.1f}'
            ],
            'Meaning': [
                'Thermal stress indicator',
                'Mechanical power (proxy)',
                'Load-weighted wear'
            ]
        }), hide_index=True, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown('---')
st.caption(
    'Model: Gradient Boosting | ROC-AUC: 0.9746 | Recall: 0.8824 | '
    'Vineeth Muraleedharan'
)
