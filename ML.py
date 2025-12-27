import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

st.title("🤖 Machine Learning App")
st.write("Upload your dataset📲")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv","xlsx"])
if uploaded_file is not None:
   
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        st.write("Number of rows and columns:", df.shape)
        st.write("Number of null values:", df.isnull().sum().sum())


        st.write("Select Target Variable:")
        target = st.selectbox("Choose the target variable", df.columns)
        X = df.drop(columns=[target])
        y = df[target]

        x = st.selectbox("Choose X variable", X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(X[[x]], y,
                                                            test_size=0.2,
                                                            random_state=42)
        st.write("Training and testing data prepared.")
        
        st.write(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
        st.success("You can now proceed with your machine learning tasks!")
        
        st.subheader("Visualizations 📊")
        # store the selection in a variable
plot_type = st.selectbox("Choose a plot type", ["Scatter Plot", "Histogram", "Box Plot"])

if st.button("Generate Plot"):
    if plot_type == "Scatter Plot":
        st.scatter_chart(df[[x, target]], x_label=x, y_label=target)
        if st.success("Scatter Plot generated."):
             st.balloons()
    elif plot_type == "Histogram":
        st.bar_chart(df[x].value_counts(), x_label=x, y_label="Count")
        if st.success("Histogram generated."):
                st.balloons()
    elif plot_type == "Box Plot":
        st.write(df[[x, target]].describe(),x_label=x, y_label=target)
        if st.success("Box Plot generated"):
                st.balloons()

        