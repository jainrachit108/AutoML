import streamlit as st
import plotly.express as px 

import os
import pandas as pd

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup, compare_models, pull, save_model, RegressionExperiment
from pycaret.classification import setup, compare_models, pull, save_model,ClassificationExperiment
with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Auto Streamlit")
    choice = st.radio("Navigation", ["Upload", "Profile",'Plot youself' ,"Model Training using pycaret", "Download"])


if os.path.exists("dataset.csv"): 
    df = pd.read_csv('dataset.csv', index_col=None)


if choice == "Upload": 
    st.title("Upload")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.success('Dataset Uploaded Successfully')
        st.dataframe(df)
        st.header('Data Statistics')
        st.write(df.describe())
        
if choice == 'Plot youself':
    st.header('Scatter Plot')
    X_axis = st.selectbox('X axis',df.columns)
    Y_axis = st.selectbox('Y axis', df.columns)
    plot = px.scatter(df, x = X_axis, y = Y_axis)
    st.plotly_chart(plot)       
        
    
    
if choice == "Profile": 
    st.title("Profile")
    profile_df = df.profile_report()
    st_profile_report(profile_df)
    


if choice == "Model Training using pycaret": 
    st.title('Train your Model')
    st.markdown('**Pycaret is an open-source, low-code machine learning library in Python that helps data scientists and machine learning practitioners to train, test, deploy, and interpret machine learning models easily and quickly.**')
    target = st.selectbox("Choose the Target", df.columns)
    model_type = st.selectbox('Choose type of model',('Classification', 'Regression'))
    
    if st.button("Run Modelling"):
        if model_type=='Classification':
            exp = ClassificationExperiment()
            
            exp.setup(df, target=target)
            setup_df = exp.pull()
            st.dataframe(setup_df)
            best_model = exp.compare_models()
            compare_df = exp.pull()
            exp.save_model(best_model, 'best_model')
            st.dataframe(compare_df)
            exp.plot_model(best_model, plot = 'confusion_matrix')
            exp.plot_model(best_model, plot = 'feature')
            
        if model_type == 'Regression':
            exp = RegressionExperiment()
            exp.setup(df, target=target)
            setup_df = exp.pull()
            st.dataframe(setup_df)
            best_model = exp.compare_models()
            compare_df = exp.pull()
            exp.save_model(best_model, 'best_model')
            st.dataframe(compare_df)
            

if choice == "Download":
    with open("best_model.pkl", 'rb') as f: 
        st.download_button("Download Model", f, "best_model_test.pkl")