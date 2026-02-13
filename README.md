**AQI Predictor Karachi**

This project predicts Air Quality Index (AQI) levels for Karachi using historical and real-time air pollution data.
The goal was to build an end-to-end ML pipeline from data collection to deployment.

This project was done as part of my internship to understand data pipelines, feature engineering, model training, and deployment.

**Project Overview**

The workflow of this project is:

- Fetch AQI data using API
- Store raw and processed data in MongoDB
- Perform EDA and feature engineering
- Train three machine learning models
- Save trained model and features
- Deploy the app using Streamlit
- Automate the pipeline using GitHub Actions (CI/CD)

**Data & Feature Engineering**

- AQI data is collected using a public API (Karachi)
- Data is stored in MongoDB
- Feature engineering is done in Python using:
  Pandas, NumPy.
- Processed data is saved so it can be reused without fetching again. Whenever required, data is loaded directly from MongoDB

**Model Training**

- Historical AQI data is used for training
- Multiple features related to air pollutants are used
- The trained model is saved using joblib
- Model performance is evaluated before deployment

**Web Application (Streamlit)**

The frontend is built using Streamlit
Users can view AQI predictions, check AQI category levels. The app is connected to the trained ML model and hosted using GitHub integration

**CI/CD (GitHub Actions)**

GitHub Actions is used to automate:
- Code checks
- Model pipeline execution
- Every push triggers the workflow
- Ensures the project runs correctly after updates

**Tech Stack**

Python (Pandas, NumPy, Scikit-learn)  
MongoDB  
Streamlit  
GitHub Actions  

**How to Run Locally**

- Clone the repository  
  git clone https://github.com/Aleenaamir512/aqi_forecast_ml

- Install dependencies  
pip install -r requirements.txt

- Run the Streamlit app  
  streamlit run app.py

**How to Run Publicly**

Click the link below to access the deployed AQI Prediction Web App:
🔗 https://aqipredictorbyaleena.streamlit.app/

**Project Report**

The detailed project report explaining data collection, EDA, feature engineering, model training and deployement is available in the 'docs' folder. 
