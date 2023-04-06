import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models
model = pickle.load(open('big_mart_model.pkl', 'rb'))
# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Financial Inclusion', #Title of the OptionMenu
                          
                          ['Financial Inclusion'], #You can add more options to the sidebar
                          icons=['cash'], #BootStrap Icons - Add more depending on the number of sidebar options you have.
                          default_index=0) #Default side bar selection

    
# bank accout Prediction Page
if (selected == 'Financial Inclusion'):
    
    # page title
    st.title('Financial Inclusion')

# getting the input data from the user
    col1, col2 = st.columns(2)
    
    with col1:
        age_of_respondent = st.number_input('age_of_respondent', min_value=0.00, max_value=90.00, step=1.00)

    with col1:
        education_level = st.selectbox('education_level', ['No formal education', 'Other/Dont know/RTA', 'Primary education', 'Secondary education', 'Tertiary education', 'Vocational/Specialised training'])
    
    with col1:
        cellphone_access = st.selectbox('cellphone_access', ['Yes', 'No'])

    with col2:
        job_type = st.selectbox('job_type', ['Dont Know/Refuse to answer','Farming and Fishing','Formally employed Government','Formally employed Private','Government Dependent','Informally employed','No Income','Other Income','Remittance Dependent','Self employed'])

    with col2:
        location_type = st.selectbox('location_type', ['Rural', 'Urban'])