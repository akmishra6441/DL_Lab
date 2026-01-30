# Build a Logistic Regression model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc). in Titanic dataset. 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
# Using seaborn's built-in dataset for convenience
titanic_df = sns.load_dataset('titanic')

print("Original Titanic dataset head:")
print(titanic_df.head())
print("\nMissing values before preprocessing:")
print(titanic_df.isnull().sum())

# --- Feature Engineering and Preprocessing ---

# Drop columns that are not useful for the model or have too many missing values
titanic_df = titanic_df.drop(['embark_town', 'deck', 'alone', 'adult_male'], axis=1)

# Handle missing values for 'Age': Fill with the median age
titanic_df['age'].fillna(titanic_df['age'].median(), inplace=True)

# Handle missing values for 'Embarked': Fill with the most frequent port
titanic_df['embarked'].fillna(titanic_df['embarked'].mode()[0], inplace=True)

# Convert categorical features into numerical using one-hot encoding
# 'sex', 'embarked', 'class' (Pclass is already numerical, but 'class' is categorical)
titanic_df = pd.get_dummies(titanic_df, columns=['sex', 'embarked', 'class'], drop_first=True)

# Drop original 'who' and 'pclass' columns if 'class' is used as categorical
titanic_df = titanic_df.drop(['who', 'pclass'], axis=1)

# Drop 'alive' column as it's redundant with 'survived'
titanic_df = titanic_df.drop('alive', axis=1)

# Drop 'fare' and 'parch' for now to simplify the model, we can add them back later if needed
titanic_df = titanic_df.drop(['fare', 'parch'], axis=1)

print("\nProcessed Titanic dataset head:")
print(titanic_df.head())
print("\nMissing values after preprocessing:")
print(titanic_df.isnull().sum())

# Define features (X) and target (y)
X = titanic_df.drop('survived', axis=1)
y = titanic_df['survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Initialize and train the Logistic Regression model
log_reg_model = LogisticRegression(solver='liblinear', random_state=42)
log_reg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Interpret Model Coefficients to answer the question ---
print("\nModel Coefficients (Impact on Log-Odds of Survival):")
coefficients_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg_model.coef_[0],
    'Abs_Coefficient': np.abs(log_reg_model.coef_[0])
})
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False)
print(coefficients_df)

print("\nInterpretation of Coefficients:")
print("- Positive coefficients indicate features that increase the likelihood of survival.")
print("- Negative coefficients indicate features that decrease the likelihood of survival.")
print("\nBased on the coefficients:")
for index, row in coefficients_df.iterrows():
    if row['Coefficient'] > 0:
        print(f"  - A higher '{row['Feature']}' value or being in this category increases the odds of survival.")
    else:
        print(f"  - A higher '{row['Feature']}' value or being in this category decreases the odds of survival.")
