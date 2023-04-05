{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0b5084f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder\n",
    "\n",
    "\n",
    "import pickle\n",
    "import streamlit as st\n",
    "from streamlit_option_menu import option_menu\n",
    "\n",
    "\n",
    "# loading the saved models\n",
    "model = pickle.load(open('C:/Users/Al Amin/Downloads/NITDA_AI_Class-main/NITDA_AI_Class-main/ML_Compete/Classification/big_mart_model.pkl', 'rb'))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
