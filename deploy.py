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
                          
                          ['Financial Inclusion','Big Mart Sales Prediction'], #You can add more options to the sidebar
                          icons=['shop', 'cash'], #BootStrap Icons - Add more depending on the number of sidebar options you have.
                          default_index=0) #Default side bar selection

    
# bank accout Prediction Page
if (selected == 'Financial Inclusion'):
    
    # page title
    st.title('Financial Inclusion')

# getting the input data from the user
    col1, col2 = st.columns(2)
    
    with col1:
        Item_Visibility = st.number_input('Item Visibility', min_value=0.00, max_value=0.40, step=0.01)

    with col1:
        Item_MRP = st.number_input('Item MRP', min_value=30.00, max_value=270.00, step=1.00)
    