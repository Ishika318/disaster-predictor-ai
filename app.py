import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Disaster Prediction AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .prediction-box {
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 18px 14px;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    h1, h2, h3 {
        letter-spacing: 0.3px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = {}
    base = "models/"

    # Cyclone CNN Model
    try:
        models["cyclone"] = tf.keras.models.load_model(base + "cyclone_cnn_model.keras")
    except Exception as e:
        models["cyclone_error"] = str(e)

    try:
        models["flood"] = joblib.load(base + "flood_model.pkl")
        models["flood_feat"] = joblib.load(base + "flood_features.pkl")
    except: pass
    try:
        models["landslide"] = joblib.load(base + "landslide_model.pkl")
        models["landslide_feat"] = joblib.load(base + "landslide_features.pkl")
    except: pass
    try:
        models["earthquake"] = joblib.load(base + "earthquake_model.pkl")
        models["earthquake_feat"] = joblib.load(base + "earthquake_features.pkl")
        models["earthquake_le"] = joblib.load(base + "earthquake_le_region.pkl")
    except: pass
    return models

models = load_models()

st.sidebar.markdown("## Disaster Prediction AI")
st.sidebar.markdown("<p style='color:gray; font-size:0.85rem; margin-top:-10px;'>AI-powered early warning system</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "Home",
    "Cyclone",
    "Flood",
    "Landslide",
    "Earthquake"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Team:**")
st.sidebar.markdown("OmmkarKhandai")
st.sidebar.markdown("Ishika318")

if page == "Home":
    st.markdown("<h1 style='text-align:center; color:#e74c3c; letter-spacing:1px;'>DISASTER PREDICTION AI</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:gray; font-weight:400;'>AI-powered early warning system for natural disasters</h4>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cyclone", "99.9%", "CNN Model")
    col2.metric("Flood", "87%", "Random Forest")
    col3.metric("Landslide", "80.7%", "Random Forest")
    col4.metric("Earthquake", "68.4%", "Random Forest")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### About
        This AI system predicts 4 types of natural disasters:
        - **Cyclone** — Satellite image (CNN) + Weather parameters
        - **Flood** — Environmental factors
        - **Landslide** — Terrain & geological data
        - **Earthquake** — Seismic data analysis
        """)
    with col2:
        st.markdown("""
        ### Technologies
        - Python, Scikit-learn, TensorFlow
        - Pandas, NumPy, Plotly
        - Streamlit deployment
        - CNN, Random Forest models
        - Real USGS & INSAT3D data
        """)

    fig = px.bar(
        x=["Cyclone CNN", "Flood", "Landslide", "Earthquake"],
        y=[99.9, 87.0, 80.7, 68.4],
        color=["Cyclone CNN", "Flood", "Landslide", "Earthquake"],
        color_discrete_sequence=["#3498db","#2ecc71","#e74c3c","#f39c12"],
        labels={"x":"Disaster","y":"Accuracy (%)"},
        title="Model Accuracy"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Cyclone":
    st.title("Cyclone Prediction")
    st.markdown("CNN-based satellite image detection **and** weather parameter-based risk assessment.")

    tab1, tab2 = st.tabs(["Satellite Image (CNN)", "Weather Parameters"])

    # ---------------- TAB 1: CNN IMAGE PREDICTION ----------------
    with tab1:
        st.subheader("Upload Satellite Image")
        uploaded_file = st.file_uploader(
            "Upload Satellite Image",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=400)

            img = image.resize((128, 128))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            if st.button("Predict from Image", use_container_width=True):
                if "cyclone" in models:
                    raw_pred = models["cyclone"].predict(img)[0][0]
                    pred = 1 - raw_pred # labels were inverted during training, so flip here
                    if pred > 0.5:
                        st.error(f"Cyclone Detected\n\nConfidence: {pred*100:.2f}%")
                    else:
                        st.success(f"No Cyclone\n\nConfidence: {(1-pred)*100:.2f}%")

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred*100,
                        title={"text": "Cyclone Probability (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "red"},
                            "steps": [
                                {"range": [0, 40], "color": "rgba(46,204,113,0.3)"},
                                {"range": [40, 70], "color": "rgba(243,156,18,0.3)"},
                                {"range": [70, 100], "color": "rgba(231,76,60,0.3)"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Cyclone CNN model not loaded!")
                    if "cyclone_error" in models:
                        st.code(models["cyclone_error"])
        else:
            st.info("Upload a satellite image to get CNN-based prediction.")
            if "cyclone_error" not in models and "cyclone" not in models:
                pass
            if "cyclone_error" in models:
                with st.expander("Debug: Why CNN model failed to load"):
                    st.code(models["cyclone_error"])

    # ---------------- TAB 2: WEATHER SLIDER PREDICTION ----------------
    with tab2:
        st.subheader("Enter Weather Parameters")
        col1, col2 = st.columns(2)
        with col1:
            sst = st.slider("Sea Surface Temperature (°C)", 20.0, 30.0, 26.0)
            pres = st.slider("Atmospheric Pressure (hPa)", 980.0, 1025.0, 1000.0)
            hum = st.slider("Humidity (%)", 30.0, 100.0, 70.0)
            wind = st.slider("Wind Shear (m/s)", 5.0, 30.0, 15.0)
        with col2:
            lat = st.slider("Latitude", 5.0, 35.0, 15.0)
            depth = st.slider("Ocean Depth (m)", 50.0, 5000.0, 500.0)
            prox = st.slider("Proximity to Coastline", 0.5, 2.0, 1.0)
            pre_dist = st.selectbox("Pre-existing Disturbance", [0, 1])

        if st.button("Predict Weather Risk", use_container_width=True):
            score = 0
            if sst > 26.5: score += 3
            if pres < 1000: score += 3
            if hum > 70: score += 2
            if wind < 15: score += 2
            if pre_dist == 1: score += 4
            if depth < 500: score += 2

            if score >= 10:
                st.error("HIGH RISK — Cyclone likely!")
            elif score >= 6:
                st.warning("MEDIUM RISK — Monitor closely")
            else:
                st.success("LOW RISK — Conditions unfavorable")

            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Score", f"{score}/16")
            col2.metric("SST", "Danger" if sst > 26.5 else "Safe")
            col3.metric("Pressure", "Low" if pres < 1000 else "Normal")

elif page == "Flood":
    st.title("Flood Probability Prediction")
    st.markdown("### Enter Environmental Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        monsoon = st.slider("Monsoon Intensity", 1, 15, 5)
        topo = st.slider("Topography Drainage", 1, 15, 5)
        river = st.slider("River Management", 1, 15, 5)
        defor = st.slider("Deforestation", 1, 15, 5)
        urban = st.slider("Urbanization", 1, 15, 5)
        climate = st.slider("Climate Change", 1, 15, 5)
        dams = st.slider("Dams Quality", 1, 15, 5)
    with col2:
        silt = st.slider("Siltation", 1, 15, 5)
        agri = st.slider("Agricultural Practices", 1, 15, 5)
        encr = st.slider("Encroachments", 1, 15, 5)
        ineff = st.slider("Ineffective Disaster Preparedness", 1, 15, 5)
        drain = st.slider("Drainage Systems", 1, 15, 5)
        coastal = st.slider("Coastal Vulnerability", 1, 15, 5)
        land = st.slider("Landslides", 1, 15, 5)
    with col3:
        water = st.slider("Watersheds", 1, 15, 5)
        infra = st.slider("Deteriorating Infrastructure", 1, 15, 5)
        pop = st.slider("Population Score", 1, 15, 5)
        wetland = st.slider("Wetland Loss", 1, 15, 5)
        plan = st.slider("Inadequate Planning", 1, 15, 5)
        political= st.slider("Political Factors", 1, 15, 5)

    if st.button("Predict Flood Probability", use_container_width=True):
        if "flood" in models:
            total = monsoon+topo+river+defor+urban+climate+dams+silt+agri+encr+ineff+drain+coastal+land+water+infra+pop+wetland+plan+political
            nat_risk = monsoon+land+water
            hum_risk = defor+urban+encr
            wat_risk = river+drain+silt
            inf_risk = infra+plan

            input_data = pd.DataFrame([[monsoon,topo,river,defor,urban,climate,dams,silt,
                                        agri,encr,ineff,drain,coastal,land,water,infra,pop,
                                        wetland,plan,political,total,inf_risk,nat_risk,hum_risk,wat_risk]],
                                      columns=models["flood_feat"])
            prob = float(models["flood"].predict(input_data)[0])
            prob = max(0, min(1, prob))

            color = "#e74c3c" if prob > 0.6 else "#f39c12" if prob > 0.4 else "#2ecc71"
            label = "HIGH FLOOD RISK" if prob > 0.6 else "MEDIUM RISK" if prob > 0.4 else "LOW RISK"

            st.markdown(f"""
            <div class="prediction-box" style="background:{color}22; border:2px solid {color};">
                {label}<br><small>Flood Probability: {prob:.1%}</small>
            </div>""", unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={"text": "Flood Probability (%)"},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":color},
                       "steps":[{"range":[0,40],"color":"rgba(46,204,113,0.2)"},
                                {"range":[40,60],"color":"rgba(243,156,18,0.2)"},
                                {"range":[60,100],"color":"rgba(231,76,60,0.2)"}]}))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Flood model not loaded!")

elif page == "Landslide":
    st.title("Landslide Risk Prediction")
    st.markdown("### Enter Terrain Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        aspect = st.slider("Aspect", 1, 8, 3)
        curvature = st.slider("Curvature", 1, 8, 3)
        eq_act = st.slider("Earthquake Activity", 1, 5, 2)
        elevation = st.slider("Elevation", 1, 5, 2)
    with col2:
        flow = st.slider("Flow", 1, 5, 2)
        lithology = st.slider("Lithology", 1, 5, 2)
        ndvi = st.slider("NDVI", 1, 5, 3)
        ndwi = st.slider("NDWI", 1, 5, 2)
    with col3:
        plan_l = st.slider("Plan", 1, 8, 4)
        precip = st.slider("Precipitation", 1, 8, 4)
        profile = st.slider("Profile", 1, 5, 2)
        slope = st.slider("Slope", 1, 5, 2)

    if st.button("Predict Landslide Risk", use_container_width=True):
        if "landslide" in models:
            input_data = pd.DataFrame([[aspect, curvature, eq_act, elevation,
                                        flow, lithology, ndvi, ndwi,
                                        plan_l, precip, profile, slope]],
                                      columns=models["landslide_feat"])
            pred = models["landslide"].predict(input_data)[0]
            prob = models["landslide"].predict_proba(input_data)[0]

            label = "LANDSLIDE RISK!" if pred == 1 else "NO LANDSLIDE RISK"
            color = "#e74c3c" if pred == 1 else "#2ecc71"
            conf = prob[1]*100 if pred == 1 else prob[0]*100

            st.markdown(f"""
            <div class="prediction-box" style="background:{color}22; border:2px solid {color};">
                {label}<br><small>Confidence: {conf:.1f}%</small>
            </div>""", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", "Risk" if pred==1 else "Safe")
            col2.metric("Confidence", f"{conf:.1f}%")
            col3.metric("Slope Risk", "High" if slope > 3 else "Low")
        else:
            st.error("Landslide model not loaded!")

elif page == "Earthquake":
    st.title("Earthquake Magnitude Prediction")
    st.markdown("### Enter Seismic Parameters")

    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", -90.0, 90.0, 35.0)
        lon = st.number_input("Longitude", -180.0, 180.0, 139.0)
        depth = st.slider("Depth (km)", 0.0, 700.0, 50.0)
        rms = st.slider("Root Mean Square", 0.0, 5.0, 0.5)
    with col2:
        year = st.slider("Year", 1965, 2025, 2020)
        month = st.slider("Month", 1, 12, 6)
        region= st.selectbox("Region", ["Asia_Pacific","North_America",
                                        "South_Pacific","Middle_East","Other"])

    if st.button("Predict Magnitude", use_container_width=True):
        if "earthquake" in models:
            try:
                region_enc = models["earthquake_le"].transform([region])[0]
            except:
                region_enc = 0

            is_shallow = 1 if depth < 70 else 0
            depth_cat = 0 if depth < 70 else 1 if depth < 300 else 2

            input_data = pd.DataFrame([[lat, lon, depth, rms, year, month,
                                        region_enc, depth_cat, is_shallow,
                                        abs(lat), abs(lon), 0]],
                                      columns=models["earthquake_feat"])
            pred = models["earthquake"].predict(input_data)[0]

            colors = {"Moderate":"#f39c12","Strong":"#e67e22",
                        "Major":"#e74c3c","Great":"#8e44ad"}
            mag_info = {"Moderate":"5.5-6.0","Strong":"6.0-7.0",
                        "Major":"7.0-8.0","Great":"8.0+"}
            color = colors.get(pred, "#3498db")

            st.markdown(f"""
            <div class="prediction-box" style="background:{color}22; border:2px solid {color};">
                {pred.upper()} EARTHQUAKE<br>
                <small>Magnitude: {mag_info.get(pred,"")}</small>
            </div>""", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Category", pred)
            col2.metric("Magnitude Range", mag_info.get(pred,""))
            col3.metric("Depth Type", "Shallow" if is_shallow else "Deep")
        else:
            st.error("Earthquake model not loaded!")
