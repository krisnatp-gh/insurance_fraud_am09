import streamlit as st
import pickle
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Insurance Fraud Prediction",
    page_icon="🚗",
    layout="wide"
)

# Load model and explainer
@st.cache_resource
def load_model_and_explainer():
    with open('final_model_xgb_tuned_FIX_BGT_20251104_1140.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('lime_explainer.dill', 'rb') as f:
        explainer = dill.load(f)
    return model, explainer

# Feature unique values
categorical_features = {
    'Month': ['Dec', 'Jan', 'Oct', 'Jun', 'Feb', 'Nov', 'Apr', 'Mar', 'Aug', 'Jul', 'May', 'Sep'],
    'DayOfWeek': ['Wednesday', 'Friday', 'Saturday', 'Monday', 'Tuesday', 'Sunday', 'Thursday'],
    'Make': ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac', 'Accura', 'Dodge', 'Mercury', 
             'Jaguar', 'Nisson', 'VW', 'Saab', 'Saturn', 'Porche', 'BMW', 'Mecedes', 'Ferrari', 'Lexus'],
    'AccidentArea': ['Urban', 'Rural'],
    'DayOfWeekClaimed': ['Tuesday', 'Monday', 'Thursday', 'Friday', 'Wednesday', 'Saturday', 'Sunday', '0'],
    'MonthClaimed': ['Jan', 'Nov', 'Jul', 'Feb', 'Mar', 'Dec', 'Apr', 'Aug', 'May', 'Jun', 'Sep', 'Oct', '0'],
    'Sex': ['Female', 'Male'],
    'MaritalStatus': ['Single', 'Married', 'Widow', 'Divorced'],
    'Fault': ['Policy Holder', 'Third Party'],
    'PolicyType': ['Sport - Liability', 'Sport - Collision', 'Sedan - Liability', 'Utility - All Perils', 
                   'Sedan - All Perils', 'Sedan - Collision', 'Utility - Collision', 'Utility - Liability', 
                   'Sport - All Perils'],
    'VehicleCategory': ['Sport', 'Utility', 'Sedan'],
    'VehiclePrice': ['more than 69000', '20000 to 29000', '30000 to 39000', 'less than 20000', 
                     '40000 to 59000', '60000 to 69000'],
    'Days_Policy_Accident': ['more than 30', '15 to 30', 'none', '1 to 7', '8 to 15'],
    'Days_Policy_Claim': ['more than 30', '15 to 30', '8 to 15', 'none'],
    'PastNumberOfClaims': ['none', '1', '2 to 4', 'more than 4'],
    'AgeOfVehicle': ['3 years', '6 years', '7 years', 'more than 7', '5 years', 'new', '4 years', '2 years'],
    'AgeOfPolicyHolder': ['26 to 30', '31 to 35', '41 to 50', '51 to 65', '21 to 25', '36 to 40', 
                          '16 to 17', 'over 65', '18 to 20'],
    'PoliceReportFiled': ['No', 'Yes'],
    'WitnessPresent': ['No', 'Yes'],
    'AgentType': ['External', 'Internal'],
    'NumberOfSuppliments': ['none', 'more than 5', '3 to 5', '1 to 2'],
    'AddressChange_Claim': ['1 year', 'no change', '4 to 8 years', '2 to 3 years', 'under 6 months'],
    'NumberOfCars': ['3 to 4', '1 vehicle', '2 vehicles', '5 to 8', 'more than 8'],
    'BasePolicy': ['Liability', 'Collision', 'All Perils']
}

# Main app
def main():
    st.title("🚗 Insurance Fraud Detection System")
    st.markdown("---")
    
    try:
        model, explainer = load_model_and_explainer()
    except Exception as e:
        st.error(f"Error loading model or explainer: {str(e)}")
        return
    
    # Create input form
    st.markdown("### 📋 Enter Claim Information")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    # Dictionary to store all inputs
    inputs = {}
    
    # Group 1: Personal Information
    with col1:
        st.markdown("#### 👤 Personal Information")
        inputs['Age'] = st.number_input("Age", min_value=0, max_value=80, value=35, step=1)
        inputs['Sex'] = st.selectbox("Sex", categorical_features['Sex'])
        inputs['MaritalStatus'] = st.selectbox("Marital Status", categorical_features['MaritalStatus'])
        inputs['AgeOfPolicyHolder'] = st.selectbox("Age Group of Policy Holder", categorical_features['AgeOfPolicyHolder'])
        st.markdown("")
        
        # Group 2: Accident Details
        st.markdown("#### 🚨 Accident Details")
        inputs['Month'] = st.selectbox("Month of Accident", categorical_features['Month'])
        inputs['DayOfWeek'] = st.selectbox("Day of Week (Accident)", categorical_features['DayOfWeek'])
        inputs['WeekOfMonth'] = st.number_input("Week of Month (Accident)", min_value=1.0, max_value=5.0, value=2.0, step=1.0)
        inputs['AccidentArea'] = st.selectbox("Accident Area", categorical_features['AccidentArea'])
        inputs['Fault'] = st.selectbox("Fault", categorical_features['Fault'])
        inputs['PoliceReportFiled'] = st.selectbox("Police Report Filed?", categorical_features['PoliceReportFiled'])
        inputs['WitnessPresent'] = st.selectbox("Witness Present?", categorical_features['WitnessPresent'])
        st.markdown("")
        
        # Group 3: Claim Details
        st.markdown("#### 📄 Claim Details")
        inputs['MonthClaimed'] = st.selectbox("Month Claimed", categorical_features['MonthClaimed'])
        inputs['DayOfWeekClaimed'] = st.selectbox("Day of Week (Claimed)", categorical_features['DayOfWeekClaimed'])
        inputs['WeekOfMonthClaimed'] = st.number_input("Week of Month (Claimed)", min_value=1.0, max_value=5.0, value=2.0, step=1.0)
        inputs['Days_Policy_Claim'] = st.selectbox("Days Between Policy and Claim", categorical_features['Days_Policy_Claim'])
        inputs['PastNumberOfClaims'] = st.selectbox("Past Number of Claims", categorical_features['PastNumberOfClaims'])
        inputs['NumberOfSuppliments'] = st.selectbox("Number of Supplements", categorical_features['NumberOfSuppliments'])
    
    with col2:
        # Group 4: Policy Information
        st.markdown("#### 📋 Policy Information")
        inputs['PolicyNumber'] = st.number_input("Policy Number", min_value=1, max_value=15420, value=5000, step=1)
        inputs['PolicyType'] = st.selectbox("Policy Type", categorical_features['PolicyType'])
        inputs['BasePolicy'] = st.selectbox("Base Policy", categorical_features['BasePolicy'])
        inputs['Deductible'] = st.number_input("Deductible", min_value=300, max_value=700, value=400, step=50)
        inputs['Days_Policy_Accident'] = st.selectbox("Days Between Policy and Accident", categorical_features['Days_Policy_Accident'])
        inputs['AddressChange_Claim'] = st.selectbox("Address Change Before Claim", categorical_features['AddressChange_Claim'])
        inputs['Year'] = st.number_input("Year", min_value=1994, max_value=1996, value=1994, step=1)
        inputs['AgentType'] = st.selectbox("Agent Type", categorical_features['AgentType'])
        inputs['RepNumber'] = st.number_input("Rep Number", min_value=1, max_value=16, value=8, step=1)
        st.markdown("")
        
        # Group 5: Vehicle Information
        st.markdown("#### 🚗 Vehicle Information")
        inputs['Make'] = st.selectbox("Vehicle Make", categorical_features['Make'])
        inputs['VehicleCategory'] = st.selectbox("Vehicle Category", categorical_features['VehicleCategory'])
        inputs['VehiclePrice'] = st.selectbox("Vehicle Price Range", categorical_features['VehiclePrice'])
        inputs['AgeOfVehicle'] = st.selectbox("Age of Vehicle", categorical_features['AgeOfVehicle'])
        inputs['NumberOfCars'] = st.selectbox("Number of Cars", categorical_features['NumberOfCars'])
        inputs['DriverRating'] = st.number_input("Driver Rating", min_value=1.0, max_value=4.0, value=2.0, step=1.0)
    
    st.markdown("---")
    
    # Center the predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔍 Predict Fraud", use_container_width=True, type="primary")
    
    if predict_button:
        # Create DataFrame from inputs
        input_df = pd.DataFrame([inputs])
        
        # Make prediction
        try:
            with st.spinner("Making prediction..."):
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            # Display prediction
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if prediction[0] == 1:
                    st.error("⚠️ **FRAUDULENT CLAIM DETECTED**")
                    fraud_prob = prediction_proba[0][1] * 100
                    st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
                else:
                    st.success("✅ **LEGITIMATE CLAIM**")
                    legit_prob = prediction_proba[0][0] * 100
                    st.metric("Legitimate Probability", f"{legit_prob:.2f}%")
            
            st.markdown("---")
            
            # Generate LIME explanation
            st.markdown("### 🔍 LIME Explanation")
            st.markdown("Understanding which features contributed to this prediction:")
            
            try:
                with st.spinner("Generating explanation..."):
                    # Transform input data using preprocessing component
                    preprocessor = model.named_steps['preprocessing']
                    transformed_data = preprocessor.transform(input_df)
                    
                    # Convert to 1D array
                    transformed_1d = transformed_data.iloc[0]
                    
                    # Generate LIME explanation
                    explanation = explainer.explain_instance(
                        transformed_1d.values,
                        model['model'].predict_proba,
                        num_features=10
                    )
                    
                    # Create and display LIME plot
                    fig = explanation.as_pyplot_figure()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
            except Exception as e:
                st.error(f"Error generating LIME explanation: {str(e)}")
                st.info("The model made a prediction, but the explanation could not be generated.")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check your input values and try again.")

if __name__ == "__main__":
    main()