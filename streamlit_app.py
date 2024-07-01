# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# import streamlit as st

# medical_df = pd.read_csv('insurance.csv')

# medical_df.replace({'sex':{'male':0,'female':1}},inplace=True)
# medical_df.replace({'smoker':{'yes':0,'no':1}},inplace=True)
# medical_df.replace({'region':{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace=True)


# X= medical_df.drop('charges',axis=1)
# y = medical_df['charges']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
# lg = LinearRegression()
# lg.fit(X_train,y_train)
# y_pred = lg.predict(X_test)
# r2_score(y_test,y_pred)

# #web app
# st.title("Medical Insurance Prediction Model")
# input_text = st.text_input("Enter Person All Feature")
# input_text_splited = input_text.split(",")
# try:
#     np_df = np.asarray(input_text_splited,dtype=float)
#     print(np_df)
#     prediction = lg.predict(np_df.reshape(1,-1))
#     st.write("Medical Insurance for this person is :\n",prediction[0])
# except ValueError:
#     st.write("Please Enter numerical value")


import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st

# Load data
medical_df = pd.read_csv('insurance.csv')

# Data preprocessing
medical_df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
medical_df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
medical_df.replace({'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)

# Train-test split
X = medical_df.drop('charges', axis=1)
y = medical_df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Model training
lg = LinearRegression()
lg.fit(X_train, y_train)

# Streamlit UI
st.title("Medical Insurance Prediction")


# Input fields for all features
age_input = st.number_input("Age", min_value=0, step=1, format="%d")
sex_input = st.selectbox("Sex", ["male", "female"])
bmi_input = st.number_input("BMI  (Value from 0 to 100)")
children_input = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
smoker_input = st.selectbox("Smoker", ["yes", "no"])
region_input = st.selectbox("Region", ["southeast", "southwest", "northwest", "northeast"])

# Convert categorical variables to numerical values
sex_val = 0 if sex_input == "male" else 1
smoker_val = 0 if smoker_input == "yes" else 1
region_val = {"southeast": 0, "southwest": 1, "northwest": 2, "northeast": 3}[region_input]

# Make prediction
if st.button("Predict"):
    try:
        # Prepare input data
        input_data = np.array([age_input, sex_val, bmi_input, children_input, smoker_val, region_val]).reshape(1, -1)

        # Make prediction   
        prediction = lg.predict(input_data)
        prediction_rounded = round(prediction[0])
        print(input_data)
        # Display result
        st.write("Medical Insurance for this person is:", prediction_rounded)
    except ValueError:
        st.write("Please enter valid input values")
        



