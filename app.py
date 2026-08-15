import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_breast_cancer


MODEL_PATH = Path(__file__).with_name('breast_cancer_model.pkl')
SCALER_PATH = Path(__file__).with_name('scaler.pkl')
DATASET = load_breast_cancer()
STANDARD_FEATURES = list(DATASET.feature_names)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model


@st.cache_resource
def load_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError("scaler.pkl not found. Save your fitted StandardScaler next to app.py and then run the app again.")
    return joblib.load(SCALER_PATH)


@st.cache_data
def get_default_values():
    medians = np.median(DATASET.data, axis=0)
    return medians.tolist()


@st.cache_data
def get_feature_names(model):
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    return STANDARD_FEATURES


@st.cache_data
def get_label_mapping(model):
    return {0: 'Malignant', 1: 'Benign'}


scaler = load_scaler()


def compute_prediction(model, values):
    X = np.asarray(values, dtype=np.float32).reshape(1, -1)
    X_scaled = scaler.transform(X)
    probs = model.predict(X_scaled, verbose=0)[0]
    pred_index = int(np.argmax(probs))
    pred_label = {0: 'Malignant', 1: 'Benign'}.get(pred_index, str(pred_index))
    return pred_index, pred_label, probs

st.set_page_config(
    page_title='Breast Cancer Risk Dashboard',
    page_icon='🎗️',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #fff5f7 0%, #f3f8ff 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .metric-card {
            background: rgba(255,255,255,0.8);
            border: 1px solid rgba(140, 160, 180, 0.18);
            border-radius: 16px;
            padding: 1rem 1.25rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
        }
        h1, h2, h3 {
            letter-spacing: -0.03em;
            color: #000000;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model()
feature_names = get_feature_names(model)
label_map = get_label_mapping(model)

st.markdown("<h1 style='color: black; margin-bottom: 0.2rem;'>Breast Cancer AI Risk Assessment</h1>", unsafe_allow_html=True)
st.caption('Clinical decision-support dashboard powered by a trained breast-cancer classifier.')

with st.sidebar:
    st.header('Patient data input')
    st.caption('Tune the biopsy measurements to generate a prediction.')

    defaults = get_default_values()
    values = {}
    for i, feature in enumerate(feature_names):
        low = float(np.min(DATASET.data[:, i]))
        high = float(np.max(DATASET.data[:, i]))
        value = float(defaults[i])
        values[feature] = st.slider(
            feature,
            min_value=round(low, 3),
            max_value=round(high, 3),
            value=round(value, 3),
            step=0.01,
            key=f'feature_{i}',
        )

    st.markdown('---')
    if st.button('Run assessment', type='primary'):
        st.session_state['run_prediction'] = True

    st.markdown('---')
    st.write('Educational use only. Not a substitute for specialist review.')

if 'run_prediction' not in st.session_state:
    st.session_state['run_prediction'] = False

if st.session_state['run_prediction']:
    arr = [values[name] for name in feature_names]
    pred_index, pred_label, probabilities = compute_prediction(model, arr)

    predicted_class_index = int(pred_index)
    predicted_probability = float(probabilities[predicted_class_index]) if predicted_class_index < len(probabilities) else float(probabilities[0])

    if pred_label == 'Malignant':
        risk_level = 'High risk'
        accent = '#d94f70'
        callout = 'This profile shows a strong malignant pattern. Prompt specialist review and additional work-up are recommended.'
    else:
        risk_level = 'Low risk'
        accent = '#18a957'
        callout = 'This profile is consistent with a benign pattern, though clinical context still matters.'

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"<div class='metric-card'><div style='color:#000000;'>Prediction</div><div style='font-size:1.8rem;font-weight:700;color:{accent};'>{pred_label}</div></div>", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"<div class='metric-card'><div style='color:#000000;'>Risk score</div><div style='font-size:1.8rem;font-weight:700;color:{accent};'>{predicted_probability * 100:.1f}%</div></div>", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"<div class='metric-card'><div style='color:#000000;'>Assessment</div><div style='font-size:1.8rem;font-weight:700;color:{accent};'>{risk_level}</div></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='metric-card' style='margin-top:1rem; color:#000000;'><strong style='color:#000000;'>Clinical summary:</strong> <span style='color:#000000;'>{callout}</span></div>", unsafe_allow_html=True)

    st.write('')

    left_col, right_col = st.columns([1.2, 0.8])
    with left_col:
        st.markdown("<h3 style='color: black; margin-bottom: 0.5rem;'>Prediction probability</h3>", unsafe_allow_html=True)
        prob_labels = ['Malignant', 'Benign']
        probability_plot = go.Figure(
            data=[go.Bar(
                x=prob_labels,
                y=[float(p) for p in probabilities],
                marker=dict(color=['#ef476f', '#06d6a0'][:len(probabilities)]),
                text=[f'{p * 100:.1f}%' for p in probabilities],
                textposition='auto',
            )]
        )
        probability_plot.update_layout(
            template='plotly_white',
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title='Class',
            yaxis_title='Probability',
            yaxis_range=[0, 1],
        )
        st.plotly_chart(probability_plot, use_container_width=True)

    with right_col:
        st.markdown("<h3 style='color: black; margin-bottom: 0.5rem;'>Input snapshot</h3>", unsafe_allow_html=True)
        table_df = pd.DataFrame([{feature: values[feature] for feature in feature_names}])
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown("<h3 style='color: black; margin-bottom: 0.5rem;'>Feature overview</h3>", unsafe_allow_html=True)
    feature_df = pd.DataFrame({'Feature': feature_names, 'Value': [values[name] for name in feature_names]})
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.caption('Model behavior is based on the trained breast-cancer classifier saved in the project folder.')

else:
    st.info('Use the sidebar to input patient data and click “Run assessment” to generate a prediction.')
