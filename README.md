# TrustLevel's submisstion for the SNET Hackathon

Let me first explain the key components of our sultion and how they work together:

1. Compontent: Sleep Quality Prediction Model
- Uses a Bayesian network model (implemented with PyMC) to predict your sleep quality for the next night. 
- Takes user inputs (we can add more paramters if we want)
  - Personal factors (age, gender, BMI)
  - Sleep-related factors (sleep duration, resting heart rate)
  - Behavioral factors (stress level, blue light exposure)
- Outputs:
  - Probability distribution across sleep quality categories (Poor/Moderate/Good)
  - Confidence metrics

2. Component: Risk-Aware Assessment Integration
- After collecting user feedback (actual sleep quality), we:
  - Batch predictions together (for demonstration: 10 samples)
  - Send to Photrek's service through SingularityNET
  - Get back ADR metrics:
    - Accuracy: How well probabilities match actual outcomes
    - Decisiveness: Confidence in decisions
    - Robustness: Performance on edge cases

3. Component: Calibration Process
- Uses ADR metrics to:
  - Adjust model concentration factor (controls prediction confidence)
  - Update CPT (Conditional Probability Table) weights
  - Generate insights about model performance
  - Recommend model improvements

4. Data Flow:
-> See system-architecture-final.mermaid file
```
a. Prediction Flow:
User Input -> Preprocessing (data validation) -> Categorization (preparing for model) -> Bayesian Model -> Sleep Quality Prediction

b. Feedback Flow:
User Feedback -> Storage -> Batch Collection -> Photrek Evaluation

c. Calibration Flow:
ADR Metrics -> Calibration Manager -> Model Updates -> Improved Predictions
```

This architecture provides several advantages:
1. Continuous Improvement: Model gets better with user feedback
2. Transparency: Clear insights into model performance
3. Scalability: Modular design allows easy updates, new parameters can be easily added

---

## Installation (for Mac)

1. Clone the repository:
```bash
git clone https://github.com/trustlevel/snet_hackathon.git
cd snet_hackathon
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with the following variables:
```
SNET_PRIVATE_KEY=your_ethereum_private_key
SNET_ETH_ENDPOINT=your_ethereum_endpoint
```

## Project Structure

```
risk-aware-assessment-app/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── models/           # ML models and prediction logic
│   ├── utils/            # Utility functions
│   ├── integrations/     # SNET integration
│   └── frontend/         # Streamlit UI
├── data/                 # Data storage
└── docs/                 # Documentation
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run src/frontend/app_v5.py
```

-> Model initiation can take up 20 minutes.

To fasten the model initiation, you can change the paramters in the bayesian_model_builder_v4.py file:
- look at def _generate_initial_trace
- Change int sample to 100 (now: 500)
- Change "chains" (line 160) to 2 (now 4).

2. Access the web interface at `http://localhost:8501`

3. Calculate at leat 10 predictions with actual outcomes

4. Run Calibration Manger and Risk-Aware-Assessment:
```bash
pyhton src/scirpts/run-calibration.py
```

5. Check model changes in your terminal. It will show if the confidence level (concentration factor) will change.
