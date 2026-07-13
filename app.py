"""
app.py - Predictive Maintenance - Failure Prediction App

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title='Predictive Maintenance',
    page_icon='⚙️',
    layout='centered'
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .title {
        font-size: 2rem; font-weight: 700;
        color: #2C3E50; text-align: center;
    }
    .subtitle {
        font-size: 0.95rem; color: #7F8C8D;
        text-align: center; margin-bottom: 2rem;
    }
    .failure-box {
        background: #FDEDEC; border-left: 5px solid #E74C3C;
        border-radius: 8px; padding: 1rem;
        font-size: 1.1rem; font-weight: 600; color: #C0392B;
    }
    .safe-box {
        background: #EAFAF1; border-left: 5px solid #27AE60;
        border-radius: 8px; padding: 1rem;
        font-size: 1.1rem; font-weight: 600; color: #1E8449;
    }
    .warning-box {
        background: #FEF9E7; border-left: 5px solid #F39C12;
        border-radius: 8px; padding: 1rem;
        font-size: 1.1rem; font-weight: 600; color: #D68910;
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
    '<div class="subtitle">IIT Guwahati — Capstone Project | Predictive Maintenance</div>',
    unsafe_allow_html=True)
st.markdown('---')


# ── Quick test scenarios ──────────────────────────────────────
st.markdown('**Quick Test Scenarios:**')
col1, col2 = st.columns(2)

with col1:
    failure_btn = st.button('⚠️ Load Failure Scenario', use_container_width=True)
with col2:
    normal_btn  = st.button('✅ Load Normal Scenario',  use_container_width=True)

# Set defaults based on scenario
if failure_btn:
    default_type      = 2       # H — Heavy Duty
    default_air       = 302.0
    default_proc      = 312.0
    default_rot       = 2800
    default_torque    = 70.0
    default_tool_wear = 240
elif normal_btn:
    default_type      = 0       # L — Light Duty
    default_air       = 298.0
    default_proc      = 308.0
    default_rot       = 1500
    default_torque    = 35.0
    default_tool_wear = 50
else:
    default_type      = 1       # M — Medium Duty
    default_air       = 298.0
    default_proc      = 308.0
    default_rot       = 1550
    default_torque    = 45.0
    default_tool_wear = 150


# ── Input ─────────────────────────────────────────────────────
st.subheader('Enter Sensor Readings')

c1, c2 = st.columns(2)

with c1:
    type_options = ['L — Light Duty', 'M — Medium Duty', 'H — Heavy Duty']
    machine_type = st.selectbox(
        'Machine Type',
        type_options,
        index=default_type
    )
    type_val = {
        'L — Light Duty':  0,
        'M — Medium Duty': 1,
        'H — Heavy Duty':  2
    }[machine_type]

    air_temp = st.number_input(
        'Air Temperature (K)',
        min_value=295.0, max_value=305.0,
        value=float(default_air), step=0.1
    )

    proc_temp = st.number_input(
        'Process Temperature (K)',
        min_value=305.0, max_value=315.0,
        value=float(default_proc), step=0.1
    )

with c2:
    rot_speed = st.number_input(
        'Rotational Speed (rpm)',
        min_value=1168, max_value=2886,
        value=int(default_rot), step=10
    )

    torque = st.number_input(
        'Torque (Nm)',
        min_value=3.8, max_value=76.6,
        value=float(default_torque), step=0.1
    )

    tool_wear = st.number_input(
        'Tool Wear (min)',
        min_value=0, max_value=253,
        value=int(default_tool_wear), step=1
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

    # Predict — use 0.3 threshold for early warning
    df_input = pd.DataFrame([input_dict])[feature_names]
    scaled   = scaler.transform(df_input)
    prob     = model.predict_proba(scaled)[0][1]
    pred     = 1 if prob >= 0.3 else 0

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
            f'<div class="warning-box">🔶 EARLY WARNING — MONITOR CLOSELY<br>'
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
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    # Show computed features
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
    'Threshold: 0.3 (Early Warning) | '
    ' Vineeth Muraleedharan'
)
