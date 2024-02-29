#!/usr/bin/env python
# coding: utf-8

# # streamlit_app.py
# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
# # Load KPrototypes model
# with open(r'C:\Users\pv437\Desktop\Data Scince Folder\Projects\Project 1\Project Final\Bankruptcy_Predictor.pkl', 'rb') as f:
#     model = pickle.load(f)
# # Importing Dataset
# df = pd.read_csv(r'C:\Users\pv437\Desktop\Data Scince Folder\Projects\Project 1\bankruptcy-prevention.csv', delimiter=';')
# 
# # Renaming Features
# df.rename(columns={' management_risk':'management_risk',
#                    ' financial_flexibility':'financial_flexibility',
#                    ' credibility':'credibility',
#                    ' competitiveness':'competitiveness',
#                    ' operating_risk':'operating_risk',
#                    ' class':'class'}, inplace=True)
# 
# 
# def predict_output(features):
#     features_array = np.array(features).reshape(1, -1)
#     prediction = model.predict(features_array)
#     return prediction[0]
# 
# def validate_inputs(inputs):
#     valid_values = [0, 0.5, 1]
#     return all(val in valid_values for val in inputs)
# 
# def main():
#     # Set page background color
#     st.markdown(
#         """
#         <style>
#         body {
#             background-color: #f4f4f4;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
# 
#     # Title with background color
#     st.markdown(
#         """
#         <style>
#         .title {
#             background-color: #4CAF50;
#             color: white;
#             padding: 10px;
#             border-radius: 10px;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
#     st.markdown("<h1 class='title'>Bankruptcy Prediction App</h1>", unsafe_allow_html=True)
# 
#     # Create input widgets for the user
#     features = ['industrial_risk', 'management_risk', 'financial_flexibility', 'credibility', 'competitiveness', 'operating_risk']
#     
#     # Initialize user_inputs with None values
#     user_inputs = [None] * len(features)
# 
#     for i, feature in enumerate(features):
#         user_input = st.selectbox(f"Select input for {feature}", [None, 0.0, 0.5, 1.0])  # Include None as an option
#         user_inputs[i] = user_input
# 
#     # Check for None values in user_inputs
#     if None in user_inputs:
#         st.warning("Please fill in all input values.")
#         st.stop()
# 
#     # Create a button to trigger the prediction with background color
#     st.markdown(
#         """
#         <style>
#         .button {
#             background-color: #008CBA;
#             color: white;
#             padding: 10px;
#             border-radius: 5px;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
#     if st.button("Predict", key="predict_button"):
#         # Make prediction using the model
#         prediction = predict_output(user_inputs)
# 
#         # Display the prediction with background color
#         st.markdown(
#             """
#             <style>
#             .prediction {
#                 background-color: #008CBA;
#                 color: white;
#                 padding: 10px;
#                 border-radius: 5px;
#             }
#             </style>
#             """,
#             unsafe_allow_html=True
#         )
#         st.subheader("Model Prediction:")
#         st.markdown(f"<p class='prediction'>{'Business is Going towards Bankruptcy' if prediction == 0 else 'Business is Going towards Non-Bankruptcy'}</p>", unsafe_allow_html=True)
# 
# if __name__ == "__main__":
#     main()
# 

# In[1]:


# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load KPrototypes model
with open(r'C:\Users\pv437\Desktop\Data Scince Folder\Projects\Project 1\Project Final\Bankruptcy_Predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# Importing Dataset
df = pd.read_csv(r'C:\Users\pv437\Desktop\Data Scince Folder\Projects\Project 1\bankruptcy-prevention.csv', delimiter=';')

# Renaming Features
df.rename(columns={' management_risk':'management_risk',
                   ' financial_flexibility':'financial_flexibility',
                   ' credibility':'credibility',
                   ' competitiveness':'competitiveness',
                   ' operating_risk':'operating_risk',
                   ' class':'class'}, inplace=True)

def predict_output(features):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return prediction[0]

def main():
    # Set page background color
    st.markdown(
        """
        <style>
        body {
            background-color: #f4f4f4;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title with background color
    st.markdown(
        """
        <style>
        .title {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='title'>Bankruptcy Prediction App</h1>", unsafe_allow_html=True)

    # Create input widgets for the user
    features = ['industrial_risk', 'management_risk', 'financial_flexibility', 'credibility', 'competitiveness', 'operating_risk']
    
    # Initialize user_inputs with None values
    user_inputs = [None] * len(features)

    for i, feature in enumerate(features):
        user_input = st.selectbox(f"Select input for {feature}", [None, 0.0, 0.5, 1.0])  # Include None as an option
        user_inputs[i] = user_input

    # Check for None values in user_inputs
    if None in user_inputs:
        st.warning("Please fill in all input values.")
    else:
        # Create a button to trigger the prediction with background color
        st.markdown(
            """
            <style>
            .button {
                background-color: #008CBA;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        if st.button("Predict", key="predict_button"):
            # Make prediction using the model
            prediction = predict_output(user_inputs)

            # Display the prediction with background color
            st.markdown(
                """
                <style>
                .prediction {
                    background-color: #008CBA;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.subheader("Model Prediction:")
            st.markdown(f"<p class='prediction'>{'Business is Going towards Bankruptcy' if prediction == 0 else 'Business is Going towards Non-Bankruptcy'}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()


# In[ ]:




