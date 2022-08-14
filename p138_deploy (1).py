#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import pickle
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# In[16]:


pickle_in=open('C:/vinnu/p138.pkl','rb')
classifier=pickle.load(pickle_in)


# In[17]:


def predict_br(industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk):
    prediction=classifier.predict([[industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk]])
    print(prediction)
    if(prediction[0]==0):
        return "Bank-Ruptcy"
    else:
        return "Non-Bank-Ruptcy"
    return prediction


# In[18]:


def main():
    st.title('Bank-Ruptcy-Prediction')
    html_temp="""
    <div style="backgroud-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank-Ruptcy-Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    industrial_risk=st.text_input("industrial_risk","Type Here")
    management_risk=st.text_input("management_risk","Type Here")
    financial_flexibility=st.text_input("financial_flexibility","Type Here")
    credibility=st.text_input("credibility","Type Here")
    competitiveness=st.text_input("competitiveness","Type Here")
    operating_risk=st.text_input("operating_risk","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_br(industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk)
    st.success("The Output is {}".format(result))
    


# In[19]:


if __name__=="__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:




