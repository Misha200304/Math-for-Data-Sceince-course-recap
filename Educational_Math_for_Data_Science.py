# Educational_Math_for_Data_Science

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Math For Data Science class recap")
st.write("Upload a CSV file to begin.")

# Uploading CSV
upload_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Reading file imported
if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Descriptive statistics
    st.subheader("Dataset Info")
    st.write(df.describe())

    # Offering the user to select the columns for modeling and selecting the model
    target_col = st.selectbox("Select target (y):", df.columns)
    features_col = st.multiselect("Select features (x):", df.columns.drop(target_col)) # droping the target column so user can't select it
    model_type = st.selectbox("Select one of the models:", ["Linear Regression", "Logistic Regression"])

    # Running Model
    if st.button("Run Model"):
        st.session_state["run_model"] = True

    # Keep results visible after rerun
    if st.session_state.get("run_model", False):
        if len(features_col) == 0:
            st.error("Please select at least one feature column.")
        else:
            x = df[features_col]
            y = df[target_col]

            if model_type == "Linear Regression":
                st.info('''
                Linear regression fits a straight line that best predicts a continuous numeric value based on one or more input features.
                
                formula: y_hat = w1x1 + w2x2 + ... + b
                
                **What it does?** it finds the weights(w) and bias(b) that minimize prediction error
                
                **When to use it?**
                - Your target variable is continuous (price, score,revenue)
                - The relationships are linear
                - You want to know how each feature affects the target
                
                **When NOT to use it?**
                - The target is categorical (0/1 → use logistic regression)
                - The relationship is non-linear
                - features are highly correlated
                - your data has complex patterns''')

                # Running the Linear regression
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                prediction = model.predict(X_test) # Using X_test as the data the model never seen before

                # Model Results
                r2 = r2_score(y_test, prediction) # Comparing the actual y in the test data set with the predictions made by the model
                st.subheader("Linear Regression Results")
                st.info("""R2 measure how much of the variation in the target variable is explained by selected features
                
                - R2 = 1, perfect prediction
                - R2 = 0, models predicts no better than the average
                - R2 < 0, model is worse than guessing the mean""")
                st.write("R² Score:", r2)
                st.info('''
                Each coefficient shows how much the target changes when a feature increases by 1 unit (holding others constant)''')
                coef_df = pd.DataFrame(model.coef_.reshape(1, -1), columns=features_col)
                st.write("Coefficients:")
                st.dataframe(coef_df)
                st.info('''
                The intercept is the model’s baseline prediction when all feature values are 0''')
                st.write("Intercept:", model.intercept_)

                show_gd = st.checkbox("Want to apply Gradient Descent to minimize the loss?")

                if show_gd:
                    st.subheader("Gradient Descent Explanation & Visualization")

                    # Convert to numpy
                    X_np = X_train[features_col].to_numpy()
                    y_np = y_train.to_numpy()

                    n = X_np.shape[0]  # number of samples
                    n_features = X_np.shape[1]  # number of features

                    # Sliders
                    lr = st.slider("Learning Rate", 0.001, 0.1, 0.01)
                    epochs = st.slider("Number of Iterations", 100, 2000, 500)

                    # Initialize parameters
                    w = np.zeros(n_features)
                    b = 0
                    losses = []
                    plot_area = st.empty()

                    # GD Loop
                    for epoch in range(epochs):
                        y_pred = X_np @ w + b
                        error = y_pred - y_np

                        dw = (2 / n) * (X_np.T @ error)
                        db = (2 / n) * error.sum()

                        w = w - lr * dw
                        b = b - lr * db

                        loss = (error ** 2).mean()
                        losses.append(loss)
                        # Loss curve
                        if epoch % 5 == 0:  # update every 5 steps
                            fig, ax = plt.subplots()
                            ax.plot(losses)
                            ax.set_title("Gradient Descent Convergence")
                            ax.set_xlabel("Iteration")
                            ax.set_ylabel("Loss (MSE)")
                            plot_area.pyplot(fig)

                    # Output
                    st.write("### Gradient Descent Results")
                    st.write("Final Weights:", w)
                    st.write("Bias:", b)
                    st.write("Final Loss:", losses[-1])

                    st.info("""
                    **How Gradient Descent Works**

                    **Step 1 — Initialize parameters**
                    - Set all weights `w` to 0 (or small random numbers)
                    - Set bias `b = 0`

                    **Step 2 — Choose hyperparameters**
                    - Learning rate (how big each step is)
                    - Number of iterations (epochs)

                    **Step 3 — Make predictions**
                    `y_hat = X @ w + b`

                    **Step 4 — Compute the error**
                    `error = y_hat - y_actual`

                    **Step 5 — Compute gradients (partial derivatives)**  
                    These tell us the direction to adjust w and b:

                    - For each weight:
                      `dw = (2/n) * (X.T @ error)`  
                      - `n` = number of samples (rows)  
                      - `X.T @ error` multiplies each feature column by its corresponding error  
                      - This shows how much each feature contributed to the prediction mistake

                    - For the bias:
                      `db = (2/n) * error.sum()`

                    **Step 6 — Update parameters**
                    - `w = w - lr * dw`  
                    - `b = b - lr * db`

                    Each update reduces the loss.

                    **Step 7 — Repeat until convergence**  
                    Loss should go down each iteration, and the line fits the data better""")

            elif model_type == "Logistic Regression":
                x = df[features_col]
                y = df[target_col]

                st.info('''
                Logistic regression predicts the probability that an observation belongs to a specific class (0 or 1).  
                Instead of a straight line, it uses the sigmoid function to squeeze predictions into the 0–1 range.
                
                Formula: z = w1x1 + w2x2 + ... + b, Linear Function
                
                y_hat = 1 / (1 + e^(-z)) → sigmoid function (squzes between 0 and 1)
                
                **What it does:**  
                It finds the weights (w) and bias (b) that minimize classification error by using the log-loss   
                Higher y_hat = more likely the model predicts class 1.
                
                **When to use it:**
                - Your target variable is binary (0/1: pass/fail, churn/stay, win/loss)
                - You want probability estimates (e.g., “chance of disease = 0.87”)
                - You want interpretable weights that show how features affect the outcome
                
                **When NOT to use it:**
                - The target variable is continuous 
                - The relationship between features and target is highly non-linear''')

                # Running the Logistic regression
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                model = LogisticRegression()
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)  # Using X_test as the data the model never seen before

                # Results
                st.info('''
                Each coefficient shows how the **log-odds** changes of the target, when the feaure increases by 1
                
                - Positive coefficient → increases the probability of class 1 
                - Negative coefficient → decreases the probability of class 1''')
                coef_df = pd.DataFrame(model.coef_.reshape(1, -1), columns=features_col)
                st.write("Coefficients:")
                st.dataframe(coef_df)
                st.info('''
                The intercept shifts the overall log-odds of the prediction, represent the model's baseline, when all features are zero''')
                st.write("Intercept:", model.intercept_)

                show_log_loss = st.checkbox("Apply Log Loss to minimize the loss?")
                show_cm = st.checkbox("Show Confusion Matrix & Metrics")

                if show_log_loss:
                    st.subheader("Log Loss Explanation & Visualization")

                    # Convert to numpy
                    X_np = X_train[features_col].to_numpy()
                    y_np = y_train.to_numpy()

                    n = X_np.shape[0]  # number of samples
                    n_features = X_np.shape[1]  # number of features
                    w = np.zeros(n_features)
                    b = 0
                    losses = []

                    lr = st.slider("Learning Rate", 0.001, 0.1, 0.01)
                    epochs = st.slider("Number of Iterations", 100, 2000, 500)
                    plot_area = st.empty()

                    for epoch in range(epochs):
                        # Models prediction
                        z = X_np @ w + b # Linear form
                        p = 1/ (1+np.exp(-z)) # Applying the sigmoid to convert into probability

                        # Applying Logg Loss
                        # If actual y = 1, then loss = -In(p)
                        # If actual y = 0, then loss = -In(1-p)
                        loss = -np.mean(y_np * np.log(p) + (1-y_np) * np.log(1-p))
                        losses.append(loss)

                        # Computing partial derivatives
                        error = p - y_np
                        dw = (1/n) * (X_np.T @ error) # Muliplying metricies
                        db = (1/n) * error.sum()

                        # Updating weights
                        w = w - lr * dw
                        b = b - lr * db

                        # Visualizing
                        if epoch % 5 == 0:  # update every 5 steps
                            fig, ax = plt.subplots()
                            ax.plot(losses)
                            ax.set_title("Gradient Descent Convergence")
                            ax.set_xlabel("Iteration")
                            ax.set_ylabel("Loss (Log Loss)")
                            plot_area.pyplot(fig)

                    # Output
                    st.write("### Log Loss Results")
                    st.write("Final Weights:", w)
                    st.write("Bias:", b)
                    st.write("Final Loss:", losses[-1])

                    st.info('''
                    **How Log Loss works?**
                     - Step 1: Initialize parameters
                        - Set all weights `w` to 0 
                        - Set bias `b = 0` 
                    - Step 2: Choose hyperparameters
                        - Learning rate (how big each step is)
                        - Number of iterations (epochs)
                    - Step 3: Make predictions
                        - y_hat = X @ w + b 
                    - Step 4: Apply sigmoid
                        - Given the y_predicted transform it using sigmoid and get the output between (0,1)
                    - Step 5: Find loss
                        - Use this formulas for loss: loss = -np.mean(y_np * np.log(p) + (1-y_np) * np.log(1-p))
                    - Step 6: Calculate error
                        - compute the error: y_pred - y_actual
                    - Step 7: Calculate partial derivatives
                        - dw = (2/n) * (X.T @ error)
                        - db = (2/n) * error.sum()
                    - Step 8: Update the weights and biases
                        - w = w - lr * dw
                        - b = b - lr * db
                        - each update reduces loss''')

                elif show_cm:
                    st.subheader("Confusion Matrix and Metrics")
                    threshold = st.slider("Threshold", 0.01, 0.99, 0.5)
                    st.info('''
                    Threshold is cutting your data
                    
                    - Higher threshold, meaning we catch fewer False Positives
                    - Lower threshold, meaning we catch fewer False Negatives
                    - As we change threshold recall and precision moves in opposite direction''')

                    y_proba = model.predict_proba(X_test)[:, 1] # making prediction in the format column 0 and 1 and we write [:,1], meaning take all rows from prob of class 1
                    y_pred = (y_proba >= threshold).astype(int)

                    cm = confusion_matrix(y_test, y_pred) # compared actual from predicted
                    st.write("###Confusion Matrix")

                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("True Label")
                    st.pyplot(fig)

                    metric_preference = st.selectbox("Which mistakes matter most for you?",
                        ["Reduce False Negatives (use Recall)", "Reduce False Positives (use Precision)","Balance both (use F1 Score)"])

                    if metric_preference == "Reduce False Negatives (use Recall)":
                        recall = recall_score(y_test, y_pred)
                        st.write("Recall:", round(recall, 3))
                        st.info('''
                        Recall is best when False Negatives are expensive''')
                    elif metric_preference == "Reduce False Positives (use Precision)":
                        precision = precision_score(y_test, y_pred)
                        st.write("Precision:", round(precision, 3))
                        st.info('''
                        Precision is best when False Positives are expensive''')
                    else:
                        f1 = f1_score(y_test, y_pred)
                        st.write("F1:", round(f1, 3))
                        st.info('''
                        F1 is best when you want the balance between Recall and Precision''')



































