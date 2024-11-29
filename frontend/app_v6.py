# src/frontend/app_v6.py
# successfully ran 27.11, including correct storage of actual sleep quality
# successfully ran 27.11, with new files and updated code
# Updated 27.11: Added calibration button to sidebar

import os
import sys
from pathlib import Path

# Get the absolute path to the project root
root_dir = Path(__file__).parent.parent.absolute()

# Add the root directory to Python path
os.environ["PYTHONPATH"] = str(root_dir)
sys.path.insert(0, str(root_dir))

import streamlit as st
import plotly.graph_objects as go
from src.utils.preprocessing import preprocess_user_data
from src.models.sleep_quality_predict_v6 import SleepQualityPredict

def create_prediction_plot(predictions):
    """Create a plotly figure for prediction visualization."""
    categories = list(predictions['probabilities'].keys())
    probabilities = list(predictions['probabilities'].values())
    
    colors = {
        'Poor': '#FF9999',
        'Moderate': '#FFB366',
        'Good': '#99FF99'
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=probabilities,
            marker_color=[colors[cat] for cat in categories],
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Sleep Quality Prediction Probabilities',
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        showlegend=False,
        plot_bgcolor='white'
    )
    
    return fig

def display_prediction_results(prediction_results, model):
    """Display prediction results and feedback section."""
    st.markdown("---")
    st.subheader("Prediction Results")

    # Show visualization
    fig = create_prediction_plot(prediction_results)
    st.plotly_chart(fig, use_container_width=True)

    # Display metrics in columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Predicted Quality",
            value=prediction_results['predicted_category']
        )
    with col2:
        confidence = max(prediction_results['probabilities'].values())
        st.metric(
            label="Confidence",
            value=f"{confidence:.1%}"
        )

    # Add feedback section
    st.markdown("---")
    st.subheader("Sleep Quality Feedback")
    
    # Simple slider and button
    quality = st.slider(
        "How well did you actually sleep?",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        key='quality_slider'
    )
    
    if st.button("Submit Sleep Quality"):
        try:
            model.record_actual_quality(
                prediction_id=prediction_results['prediction_id'],
                actual_quality=quality
            )
            st.success(f"Recorded sleep quality: {quality}")
            # Debug print
            st.write("Debug - Attempting to store quality:", quality)
            st.write("Debug - For prediction:", prediction_results['prediction_id'])
        except Exception as e:
            st.error(f"Error recording feedback: {str(e)}")
            st.write("Debug - Error details:", str(e))

def initialize_model():
    """Initialize the model and store in session state."""
    if 'model' not in st.session_state:
        with st.spinner('Initializing prediction model... This may take a few minutes... (Loading Bayesian Model and generating initial predictions)'):
            st.session_state.model = SleepQualityPredict()
            st.success('Model initialized successfully!')
    return st.session_state.model

def main():
    st.set_page_config(
        page_title="Sleep Quality Prediction",
        page_icon="💤",
        layout="wide"
    )
    
    # Initialize model
    model = initialize_model()


    # Initialize session state for prediction results
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None

    # Add calibration update button in sidebar
    with st.sidebar:
        if st.button("Update Model From Calibration"):
            filepath = Path(__file__).parent.parent / 'src' / 'data' / 'concentration_factor.txt'
            try:
                with open(filepath, 'r') as f:
                    calibrated_factor = float(f.readlines()[-1].strip())
                    
                # Store old factor for comparison
                old_factor = st.session_state.model.builder.concentration_factor
                
                # Create new model
                st.session_state.model = SleepQualityPredict()
                # Set new factor
                st.session_state.model.builder.concentration_factor = calibrated_factor
                # Force rebuild of model with new factor
                st.session_state.model.builder.model = None  # Clear existing model
                st.session_state.model.builder.build_model()  # Rebuild with new factor
                
                st.success(f"Model updated! Concentration factor: {old_factor:.3f} → {calibrated_factor:.3f}")
                st.info("Model rebuilt with new calibration parameters")
            except Exception as e:
                st.error(f"Could not load calibrated model parameters: {str(e)}")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            margin-top: 20px;
        }
        .stSubheader {
            margin-bottom: 20px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("Sleep Quality Prediction")
    
    # Create description
    st.markdown("""
        This tool predicts sleep quality based on various personal and behavioral factors.
        Fill in the form below to get your prediction.
    """)
    
    # Create form for input collection
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal Information")
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
                help="Select your gender"
            )
            
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=120,
                value=30,
                step=1,
                help="Enter your age (18-120 years)"
            )
            
            weight = st.number_input(
                "Weight (kg)",
                min_value=30.0,
                max_value=300.0,
                value=70.0,
                step=0.1,
                format="%.1f",
                help="Enter your weight in kilograms (30-300 kg)"
            )
            
            height = st.number_input(
                "Height (m)",
                min_value=1.0,
                max_value=2.5,
                value=1.7,
                step=0.01,
                format="%.2f",
                help="Enter your height in meters (1.0-2.5 m)"
            )

        with col2:
            st.subheader("Sleep-Related Factors")
            resting_heart_rate = st.number_input(
                "Resting Heart Rate (bpm)",
                min_value=30,
                max_value=150,
                value=70,
                step=1,
                help="Enter your resting heart rate (30-150 bpm)"
            )
            
            sleep_duration = st.number_input(
                "Average Sleep Duration (hours)",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.1,
                format="%.1f",
                help="Enter your typical sleep duration (0-24 hours)"
            )
            
            stress_level = st.slider(
                "Stress Level",
                min_value=0,
                max_value=10,
                value=5,
                step=1,
                help="Rate your stress level (0 = None, 10 = Extreme)"
            )
            
            blue_light_hours = st.number_input(
                "Hours Before Bed Without Blue Light",
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.1,
                format="%.1f",
                help="Hours before bed without screen exposure (0-3 hours)"
            )

        # Submit button
        submitted = st.form_submit_button("Predict Sleep Quality")

    # Handle prediction when form is submitted
    if submitted:
        try:
            # Prepare input data
            raw_data = {
                "gender": gender,
                "age": float(age),
                "weight": float(weight),
                "height": float(height),
                "resting_heart_rate": float(resting_heart_rate),
                "sleep_duration": float(sleep_duration),
                "stress_level": float(stress_level),
                "blue_light_hours": float(blue_light_hours)
            }

            # Make prediction
            with st.spinner("Generating prediction..."):
                prediction_results = model.predict(raw_data)
                st.session_state.prediction_results = prediction_results

            # Display results (will stay visible)
            display_prediction_results(prediction_results, model)

        except Exception as e:
            st.error("An error occurred during prediction")
            st.error(f"Error details: {str(e)}")

    # Always display results if they exist
    elif st.session_state.prediction_results is not None:
        display_prediction_results(st.session_state.prediction_results, model)

if __name__ == "__main__":
    main()