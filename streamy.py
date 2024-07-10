import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

st.title("c'etait difficile")
energi=pd.read_csv('Expresso_churn_dataset.csv', nrows =155000)

energi.info()

energi.isnull().sum()


for col in energi.select_dtypes(object).columns:
    m = LabelEncoder()
    energi[col]=m.fit_transform(energi[col])
    energi[col]=energi[col].astype('category')

energi.drop(['user_id','REGION','TENURE','MRG','TOP_PACK'], axis=1, inplace=True)

energi.fillna(energi.mean(), inplace=True)

energi.head()

st.dataframe(energi)
