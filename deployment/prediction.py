import pandas as pd
import numpy as np
import streamlit as st
import datetime
import pickle
from tensorflow.keras.models import load_model

# Load the combined pipeline
with open('final_pipeline.pkl', 'rb') as file_1:
  model_pipeline = pickle.load(file_1)
  
model_ann = load_model('model_seq1.h5')

def run():
    #membuat title
    st.title('Churn Prediction')
    st.subheader('Predict customer churn')
    st.markdown('---')
    st.write('Customer Data : ')
    with st.form(key='churn_customer'):
        age = st.number_input('Customer Age ? ',min_value=0,max_value=90,value=20,step=1)
        Gender = st.radio('Customer Gender ? ',('Male','Female'))
        if Gender == 'Male':
            gender = 'M'
        else:
            gender = 'F'
        region_category = st.radio('Customer Region ? ',('City','Village','Town'))
        membership_category = st.radio('Membership category : ',('No Membership','Basic Membership','Silver Membership','Gold Membership', 'Premium Membership', 'Platinum Membership'))
        joining_date = st.date_input("When customer first join date?", datetime.date(2023, 6, 1))
        joined_through_referral = st.radio('Joined through referral?',('Yes','No'))
        preferred_offer_types = st.radio('Preferred offer types?',('Without Offers', 'Credit/Debit Card Offers','Gift Vouchers/Coupons'))
        medium_of_operation = st.radio('What is the medium of customer operation?', ('Desktop', 'Smartphone', 'Both'))
        days_since_last_login = st.number_input("Days since last login", min_value=0, max_value=30, value=1, step=1,help='Put 30 if more than 30 days')
        avg_time_spent = st.number_input('Customer average time spent(in minutes)?',min_value=0,max_value=1440,value=0,step=1)
        avg_transaction_value = st.number_input('Customer average transaction value?',min_value=0,max_value=100000,value=0,step=1)
        avg_frequency_login_days = st.number_input('Customer average login days?',min_value=0,max_value=100,value=0,step=1)
        points_in_wallet = st.number_input('How much point did customer gain from transaction?', min_value=0, max_value=100, value=0, step=1)
        used_special_discount = st.radio('Are customer used special discount?',('Yes','No'))
        offer_application_preference = st.radio('Are customer prefer offers?',('Yes','No'))
        past_complaint = st.radio('Are costumer raised any complaint?',('Yes','No'))
        complaint_status = st.radio('Did the complaint resolve?',('No Information Available', 'Not Applicable', 'Unsolved', 'Solved', 'Solved in Follow-up'))
        feedback = st.radio('What is the most complaint customer give?',('Poor Website', 'Poor Customer Service', 'Too many ads', 
                                                                        'Poor Product Quality', 'No reason specified', 'Products always in Stock', 
                                                                         'Reasonable Price', 'Quality Customer Care', 'User Friendly Website'))
        
        submitted = st.form_submit_button('Predict')
        
        # dataframe
        data_inf = {
                    'age': age,
                    'gender': gender,
                    'region_category': region_category,
                    'membership_category': membership_category,
                    'joining_date': joining_date,
                    'joined_through_referral': joined_through_referral,
                    'preferred_offer_types': preferred_offer_types,
                    'medium_of_operation': medium_of_operation,
                    'days_since_last_login': days_since_last_login,
                    'avg_time_spent': avg_time_spent,
                    'avg_transaction_value': avg_transaction_value,
                    'avg_frequency_login_days': avg_frequency_login_days,
                    'points_in_wallet': points_in_wallet,
                    'used_special_discount': used_special_discount,
                    'offer_application_preference': offer_application_preference,
                    'past_complaint': past_complaint,
                    'complaint_status': complaint_status,
                    'feedback': feedback,
                    }
        
        st.write('Customer Summary')
        data_inf = pd.DataFrame([data_inf])
        st.dataframe(data_inf.T, width=800, height=495)
        
    if submitted:
        # Predict using created pipeline
        data_inf_transform = model_pipeline.transform(data_inf)
        y_pred_inf = model_ann.predict(data_inf_transform)
        y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)
        if y_pred_inf == 0:
            pred = 'Not Churn'
        else:
            pred = 'Churn'
        st.markdown('---')
        st.write('# Prediction : ', (pred))
        st.markdown('---')
        
if __name__ == '__main__':
    run()
