import streamlit as st

import eda
import prediction

navigation = st.sidebar.radio('Page : ', ('EDA', 'Predict Customer Churn'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()