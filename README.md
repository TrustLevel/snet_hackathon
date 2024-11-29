# TrustLevel's submission for the SNET Hackathon

Explaination of the our approach: https://www.loom.com/share/1d4c9d1d31454cd78a5daed73eb582ad?sid=2afe32c0-df24-4b79-9e5d-3a44003ac0e2

Let me first explain the key components of our sultion and how they work together:

1. Compontent: Sleep Quality Prediction Model
- Uses a Bayesian network model (implemented with PyMC) to predict your sleep quality for the next night. 
- Takes user inputs (the model is structure to be able to add more paramters if we want)
  - Personal factors (age, gender, BMI)
  - Sleep-related factors (sleep duration, resting heart rate)
  - Behavioral factors (stress level, blue light exposure)
- Outputs:
  - Probability distribution across sleep quality categories (Poor/Moderate/Good)
  - Confidence metrics

2. Component: Risk-Aware Assessment Integration
- After collecting user feedback (actual sleep quality), we:
  - Batch predictions together (for Hackathon demonstration: 10 samples)
  - Send to Photrek's service through SingularityNET
  - Get back ADR metrics:
    - Accuracy: How well probabilities match actual outcomes
    - Decisiveness: Confidence in decisions
    - Robustness: Performance on edge cases

3. Component: Calibration Process
- Uses ADR metrics to:
  - Adjust model concentration factor which controls prediction confidence (meaning: how confident the model is about it's predctions)
  - This way, the model learns about it's prediction and can improve accuracy of it's future predictions o
  - Could be used to update CPT (Conditional Probability Table) weights (-> this part is not part of the Hackathon Demo)
  - Generate insights about model performance and recommend model improvements

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
-> Trouble shoot: Make sure you have pymc version 5.10.4, numpy 1.24.2, pytensor 2.18.6, scipy 1.10.1!

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
│   └── data    /         # Data storage
└── frontend              # Streamlit UI
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run frontend/app_v6.py
```

-> Model initiation can take a few minutes.

2. Access the web interface at `http://localhost:8501`

3. Calculate at leat 10 predictions with actual outcomes

4. Run Calibration Manger and Risk-Aware-Assessment:
```bash
pyhton src/scripts/run_calibration.py
```
-> Trouble shoot: 
If permission is denied, try first: chmod +x src/scripts/run_calibration.py and the run the command again

What happens at calibration process: If model accuracy is low for example, it will adjust the concentration factor of the model. You can see the lastest concentration factor in src/data/concentration_factor.txt after running the calibration.

Check terminal log for Risk-Assessment result and new concentration factor. It should look like this:

![image](https://github.com/user-attachments/assets/a99e1550-6b38-4f97-8035-f56ce89f446f)


5. Go back to the Streamlit UI.
```
Press button: Update Model (on the right)
Model will use new concentration factor to recalibrate the concentration factor of the model.
```
-> Trouble shoot: If concentration factor is not found, manually create concentration_factor.txt under src/data/ and enter the new factor (e.g. 0.8). 


