import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title = 'CHURN PREDICTION' ,
    initial_sidebar_state= 'expanded',
)

def run():
    # membuat title
    st.title('CHURN PREDICTION')
    st.subheader('EXPLORATORY DATA ANALYSIS(EDA)')
    st.markdown('---')

    # menambahkan gambar
    image = Image.open('Cara-Mengurangi-Churn-Rate(1).png')
    st.image(image)
    st.write('## Background')
    st.write('''Customer churn is a very high prolem in shopping businesses both online and offline, 
                therefore to minimize the churn rate on customers, 
                modeling is carried out to predict what types of customers will churn.''')
    st.write('## Objective')
    st.write('''predict churn from customer habits and how often customers make purchases/shop at our place.''')
        
    st.write('Dataset : ')
    data = pd.read_csv('churn.csv')
    st.dataframe(data)
    st.write('Column Description')
        
    markdown_text = '''
        | Column | Description |
        | --- | --- |
        | `user_id` | ID of a customer |
        | `age` | Age of a customer |
        | `gender` | Gender of a customer |
        | `region_category` | Region that a customer belongs to |
        | `membership_category` | Category of the membership that a customer is using |
        | `joining_date` | Date when a customer became a member |
        | `joined_through referral` | Whether a customer joined using any referral code or ID |
        | `preferred_offer types` | Type of offer that a customer prefers |
        | `medium_of operation` | Medium of operation that a customer uses for transactions |
        | `internet_option` | Type of internet service a customer uses |
        | `last_visit_time` | The last time a customer visited the website |
        | `days_since_last_login` | Number of days since a customer last logged into the website |
        | `avg_time_spent` | Average time spent by a customer on the website |
        | `avg_transaction_value` | Average transaction value of a customer |
        | `avg_frequency_login_days` | Number of times a customer has logged in to the website |
        | `points_in_wallet` | Points awarded to a customer on each transaction |
        | `used_special_discount` | Whether a customer uses special discounts offered |
        | `offer_application_preference` | Whether a customer prefers offers |
        | `past_complaint` | Whether a customer has raised any complaints |
        | `complaint_status` | Whether the complaints raised by a customer was resolved |
        | `feedback` | Feedback provided by a customer |
        | `churn_risk_score` | Churn score <br><br> `0` : Not churn <br> `1` : Churn |

        ---'''
    st.markdown(markdown_text)
    cat_columns = data.select_dtypes(include = 'object').drop(['user_id','last_visit_time'],axis = 1).columns.tolist()
    num_columns = data.select_dtypes(include = np.number).drop(['churn_risk_score'], axis = 1).columns.tolist()
    st.write('Churn rate')
    fig = plt.figure(figsize=(8,8))
    plt.pie(data['churn_risk_score'].value_counts(),labels = [1,0], autopct='%.0f%%', explode=[0,0.1])
    plt.title('Churn risk score')
    st.pyplot(fig)
    st.write('''from the plot data above we know that the customer churn rate is 54%''')
    st.write('## Histogram of Categorical Columns.')
    pilihan = st.selectbox('Which categorical? ',(cat_columns))
    fig= plt.figure(figsize=(15,5))
    sns.histplot(data[pilihan],kde=True)
    st.pyplot(fig)
    st.write('''From the data above we know that :

    1. in the gender column the number of male and female customers is equal
    2. most of the feedback given is poor website, poor customer service, too many ads, and poor product quality.
    3. most complaints are not implemented
    4. most customers do not become members or become basic members
    5. most customers use special discounts
    6. most customers use all the offers given.
    7. most customers come from towns and cities.
    8. most customers join using referrals or not using referrals. ''')
    st.write('## KdePlot of Numerical Columns')
    pilihan1 = st.selectbox('Which numerical?',(num_columns))
    fig = plt.figure(figsize=(11,9))
    sns.kdeplot(data[pilihan1])
    st.pyplot(fig)
    st.write('''From the data above we know that :  
             
    1. Age is evenly distributed among customers
    2. average time spent, average transaction value, and average frequency of days visited are right skewed.''')

if __name__ == '__main__':
    run()
