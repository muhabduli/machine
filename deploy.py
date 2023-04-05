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
    