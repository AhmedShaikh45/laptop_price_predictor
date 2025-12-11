import streamlit as st
import pandas as pd
import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title('Laptop Price Predictor')

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type_name = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No','Yes'])

# Ips
ips = st.selectbox('IPS', ['No','Yes'])

# screen size
screen_size = st.number_input('Screen size')

# resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080','1366x768','1600x900',
    '3840x2160','3200x1800','2880x1800',
    '2560x1600','2560x1440','2304x1440'
])

# cpu
cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())

# harddrive
hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)', [0,128,256,512,1024])

gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())

os = st.selectbox('OS', df['os'].unique())


if st.button('Predict Price'):

    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    if screen_size == 0:
        st.error("Please enter a valid screen size")
        st.stop()

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = (((X_res**2) + (Y_res**2))**0.5) / screen_size

    query = pd.DataFrame({
        'Company':[company],
        'TypeName':[type_name],
        'Ram':[ram],
        'Weight':[weight],
        'Touchscreen':[touchscreen],
        'Ips':[ips],
        'Cpu brand':[cpu],
        'HDD':[hdd],
        'SSD':[ssd],
        'Gpu brand':[gpu],
        'os':[os],
        'X_res':[X_res],
        'Y_res':[Y_res],
        'ppi':[ppi],
        'Inches':[screen_size]
    })

    pred = pipe.predict(query)[0]
    st.success("Predicted Price: ₹ " + str(int(np.exp(pred))))
