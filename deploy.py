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
        location_type = st.selectbox('location_type', ['Rural', 'Urban'])

    with col1:
        cellphone_access = st.selectbox('cellphone_access', ['Yes', 'No'])

    with col1:
        age_of_respondent = st.number_input('age_of_respondent', min_value=0.00, max_value=90.00, step=1.00)

    with col2:
        gender_of_respondent = st.selectbox('gender_of_respondent', ['Male', 'female'])

    with col2:
        education_level = st.selectbox('education_level', ['No formal education', 'Other/Dont know/RTA', 'Primary education', 'Secondary education', 'Tertiary education', 'Vocational/Specialised training'])
    
    with col2:
        job_type = st.selectbox('job_type', ['Dont Know/Refuse to answer','Farming and Fishing','Formally employed Government','Formally employed Private','Government Dependent','Informally employed','No Income','Other Income','Remittance Dependent','Self employed'])

     #Data Preprocessing
        
    data = {
            location_type:'location_type',
            cellphone_access:'cellphone_access',
            age_of_respondent: 'age_of_respondent',
            gender_of_respondent: 'gender_of_respondent',
            education_level: 'education_level',
            job_type: 'job_type'
            }
    
    oe = OrdinalEncoder(categories = [['Dont Know/Refuse to answer','Farming and Fishing','Formally employed Government','Formally employed Private','Government Dependent','Informally employed','No Income','Other Income','Remittance Dependent','Self employed']])

    oe_ed = OrdinalEncoder(categories = [['No formal education', 'Other/Dont know/RTA', 'Primary education', 'Secondary education', 'Tertiary education', 'Vocational/Specialised training']])
    scaler = StandardScaler()
    
    def make_prediction(data):
        df = pd.DataFrame(data, index=[0])

        if df['location_type'].values == 'Rural':
            df['location_type'] = 0.0
  
        if df['location_type'].values == 'Urban':
          df['location_type'] = 1.0

        if df['cellphone_access'].values == 'No':
            df['cellphone_access'] = 0.0
  
        if df['cellphone_access'].values == 'Yes':
          df['cellphone_access'] = 1.0
        
        if df['gender_of_respondent'].values == 'Male':
            df['gender_of_respondent'] = 0.0
  
        if df['gender_of_respondent'].values == 'Female':
          df['gender_of_respondent'] = 1.0

        df['education_level'] = oe.fit_transform(df[['education_level']])

        df['job_type'] = oe_ed.fit_transform(df[['job_type']])

        df['age_of_respondent'] = StandardScaler().fit_transform(df['age_of_respondent'])

        