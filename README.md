# Project Summary: Predicting Term Deposit Subscriptions Using Machine Learning

## Objective
The goal of this project was to develop machine learning models to predict whether clients of a Portuguese bank would subscribe to a term deposit based on demographic, financial, and campaign-related data. The project aimed to provide actionable insights for marketing strategies while optimizing model performance.

## Data Information
* Data Set - Data set used in the project is in the bank-additional-full.csv file
* Data Folder: A folder is created in the Google Drive to store the data file (bank-additional-full.csv). Data file was stored in the /content/drive/MyDrive/Practical-Application-3 Google Drive Folder.
* Code File - Python Code is stored in the Module_5_Activity1.ipynb file. Google Colab is being used to run the Python code.

## Steps Taken

### Data Exploration and Cleaning:

* The dataset contained over 40,000 records with variables spanning client demographics, campaign metrics, and economic indicators.
* Replaced "unknown" values with NaN, removed irrelevant columns like duration, and normalized numerical features.

### Feature Engineering:
* Created interaction features (e.g., campaign_pdays_interaction) and binned numerical variables (e.g., age and pdays).
* Performed one-hot encoding for categorical variables and scaled numerical features to improve model interpretability.

### Model Building:

* Trained and evaluated multiple machine learning models, including Logistic Regression, KNN, Decision Tree, and SVM, using default parameters.
* Compared model performance based on accuracy, precision, recall, F1 Score, and training time.

### Optimization:
* Applied hyperparameter tuning via GridSearchCV to improve model performance.
* Refined models by selecting the top 10 most important features based on feature importance analysis using Random Forest.

### Final Model Selection:

* The Random Forest model emerged as the best-performing model, with an accuracy of 89.16% and an AUC-ROC of 77.47%.
* Optimized hyperparameters (max_depth, max_features, etc.) further enhanced performance.

## Findings

### Key Influential Features:
* Economic Indicators: nr_employed, euribor3m, and emp_var_rate were among the top predictors.
* Campaign Metrics: campaign_pdays_interaction and pdays highlighted the importance of outreach timing and frequency.
* Client History: Variables like poutcome_success and prev_success_contacts underscored the significance of past interactions.

### Model Insights:

* Logistic Regression was interpretable but struggled with recall.
* KNN provided quick predictions but required careful scaling and hyperparameter tuning.
* Decision Tree suffered from overfitting without pruning.
* Random Forest effectively handled feature interactions and delivered robust performance.

## Recommendations:
### Marketing Strategy:

* Focus on clients with favorable economic indicators (e.g., euribor3m rates) and positive past campaign outcomes (poutcome_success).
* Optimize campaign timing by leveraging insights from campaign_pdays_interaction.

## Conclusion
This project successfully developed and optimized machine learning models to predict term deposit subscriptions. The insights generated from feature importance analysis and model performance evaluations offer valuable guidance for enhancing marketing efforts and targeting high-potential clients effectively. Further refinements and deployment of the model can maximize its practical utility in real-world scenarios.


## Coding Snapshots

### Create DataFrame in Python for the Data Set

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

file_path = '/content/drive/MyDrive/Practical-Application-3/bank-additional-full.csv'
df = pd.read_csv(file_path, sep = ';')

# Display unique campaign counts
unique_campaigns = df['campaign'].nunique()
unique_campaigns

df.head()
```
<img width="1680" alt="Screenshot 2025-01-20 at 3 31 20 PM" src="https://github.com/user-attachments/assets/0359a186-5b9f-4b5c-a1a9-aeb27b1f7974" />

### Data Analysis
```python
### Data Set Column Details
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
```
```python
# General overview of the data
data_info = df.info()
```
<img width="387" alt="Screenshot 2025-01-20 at 3 43 45 PM" src="https://github.com/user-attachments/assets/16be3928-6057-49b7-aa8c-584fd1672403" />

```python
# General Description of the dataframe
data_description = df.describe()
data_description
```
<img width="1158" alt="Screenshot 2025-01-20 at 3 44 42 PM" src="https://github.com/user-attachments/assets/fb32559b-d760-4ee1-a4b7-d330a7df8986" />

```python
# Checking for missing values
missing_values = df.isnull().sum()
missing_values
```
<img width="139" alt="Screenshot 2025-01-20 at 3 41 50 PM" src="https://github.com/user-attachments/assets/b76b31a9-4b61-4093-9b26-dfdd11ff4359" />

```python
# Distribution of the target variable `y`
target_distribution = df['y'].value_counts()
target_distribution
```
<img width="125" alt="Screenshot 2025-01-20 at 3 42 26 PM" src="https://github.com/user-attachments/assets/47bdbd96-5b2d-4472-aff2-c5f15b695068" />

```python
# Summary statistics for numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
numerical_summary = df[numerical_cols].describe()
numerical_summary
```
<img width="1158" alt="Screenshot 2025-01-20 at 3 48 11 PM" src="https://github.com/user-attachments/assets/d211dddf-ba3c-497c-b340-87288a92a8d9" />

```python
# Overview of categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
categorical_summary = {col: df[col].value_counts() for col in categorical_cols}
categorical_summary
```
```python
# Correlation heatmap for numerical variables
plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Heatmap of Numerical Variables")
plt.show()
```
<img width="999" alt="Screenshot 2025-01-20 at 3 50 12 PM" src="https://github.com/user-attachments/assets/8a147c19-61b6-489f-8cf1-c8870a8eb130" />

### Data Analysis Summary
The data analysis for the dataframe df reveals several key insights:

#### General Overview:
* Total records: 41,188
* 21 columns, including 11 categorical and 10 numerical features.
* No missing values in the dataset.
  
#### Target Variable (y) Distribution:
* "no" (not subscribed): 36,548 (88.7%)
* "yes" (subscribed): 4,640 (11.3%)
  
#### Key Numerical Insights:
* Most clients were contacted only once during the campaign (campaign median = 2).
* High variance in duration (range from 0 to 4918 seconds).
* pdays indicates that most clients had not been previously contacted (999).
  
#### Correlation Analysis:
* Some numerical variables (e.g., emp.var.rate, euribor3m, nr.employed) show strong relationships, possibly indicating trends in the economy.
  
#### Categorical Variable Distribution:
* Job roles are diverse, with "admin." and "blue-collar" roles being the most common.
* Most clients are "married."
* "University degree" is the most common education level.
  
#### Additional Observations:
* The majority of contacts were made via "cellular."
* The "may" month shows the highest campaign activity.

### Data Cleaning
```python
# 1. Check and handle missing values in categorical columns
missing_categorical = df.select_dtypes(include=['object']).isnull().sum()
missing_categorical
```
<img width="143" alt="Screenshot 2025-01-20 at 3 55 33 PM" src="https://github.com/user-attachments/assets/c51d5400-1df5-4b34-a68a-15f5dca68a11" />

```python
# Replace "unknown" in certain columns with NaN for better handling
columns_with_unknown = ['job', 'marital', 'education', 'default', 'housing', 'loan']
df[columns_with_unknown] = df[columns_with_unknown].replace('unknown', np.nan)
```

```python
# 2. Recheck missing values after replacements
missing_values_after = df.isnull().sum()
missing_values_after
```
<img width="155" alt="Screenshot 2025-01-20 at 3 56 44 PM" src="https://github.com/user-attachments/assets/369a5f0e-9d63-4a59-97cc-e294ac953c50" />

```python
# 4. Removing irrelevant or redundant columns
# Based on the description, "duration" should not be used for realistic modeling.
df_cleaned = df.drop(columns=['duration'])
```
```python
# 5. Rename columns for readability
df_cleaned.rename(columns=lambda x: x.replace('.', '_'), inplace=True)
```
```python
df_cleaned.info()
```
<img width="395" alt="Screenshot 2025-01-20 at 3 58 39 PM" src="https://github.com/user-attachments/assets/30b3087c-0811-4a1b-a3e3-d8d3709d4da6" />

### Data Cleaning Summary
#### Handling Missing Values:
* Replaced "unknown" in job, marital, education, default, housing, and loan columns with NaN.
#### Dropped Columns:
* Removed the duration column as it should not be used for realistic modeling.
#### Renamed Columns:
* Adjusted column names to make them more readable by replacing periods with underscores.

### Data Visualization
```python
# Visualization 1: Marital Status Impact on Subscription
plt.figure(figsize=(8, 6))
sns.countplot(x='marital', hue='y', data=df_cleaned, palette='pastel')
plt.title("Marital Status Impact on Subscription")
plt.xlabel("Marital Status")
plt.ylabel("Count")
plt.legend(title="Subscribed", loc='upper right')
plt.show()
```
![download](https://github.com/user-attachments/assets/c9a3e279-f520-47fe-b807-1dbc59ef6298)

```python
# Visualization 2: Pair Plot of Key Numerical Variables Colored by Target Variable
sns.pairplot(df_cleaned, vars=['age', 'campaign', 'pdays', 'emp_var_rate'], hue='y', palette='husl', diag_kind='kde')
plt.suptitle("Pair Plot of Key Numerical Variables by Subscription Status", y=1.02)
plt.show()
```
![download (1)](https://github.com/user-attachments/assets/5773245a-f196-467f-8fcc-4813e974e5a8)

```python
# Visualization 3: Violin Plot for Age Distribution by Education and Target Variable
plt.figure(figsize=(12, 6))
sns.violinplot(x='education', y='age', hue='y', data=df_cleaned, split=True, inner="quart", palette='muted')
plt.title("Age Distribution by Education Level and Subscription Status")
plt.xlabel("Education Level")
plt.ylabel("Age")
plt.xticks(rotation=45)
plt.show()
```
![download (2)](https://github.com/user-attachments/assets/518c4c34-21d2-40d9-a23e-c0bfc65f3aa9)

```python
# Visualization 4: Monthly Trend of Subscription Rates as a Line Plot
monthly_data = df_cleaned.groupby('month')['y'].value_counts(normalize=True).unstack()['yes'] * 100
monthly_data = monthly_data.reindex(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
plt.figure(figsize=(10, 6))
sns.lineplot(data=monthly_data, marker='o', linewidth=2, color='blue')
plt.title("Monthly Subscription Rate Trend")
plt.xlabel("Month")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=45)
plt.show()
```
![download (3)](https://github.com/user-attachments/assets/28f7b4e4-35fb-40da-89b1-b5b1e30f4c57)

```python
# Visualization 5: Heatmap of Target Variable by Job and Contact Type
heatmap_data = pd.crosstab(df_cleaned['job'], df_cleaned['contact'], normalize='index')
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Heatmap of Contact Type by Job Role")
plt.xlabel("Contact Type")
plt.ylabel("Job Role")
plt.show()
```
![download (4)](https://github.com/user-attachments/assets/a212fe6c-a237-40a6-a07e-547b39ce8abf)

```python
# Visualization 6: Distribution of Employment Variation Rate by Housing Loan and Subscription
plt.figure(figsize=(12, 6))
sns.boxenplot(x='housing', y='emp_var_rate', hue='y', data=df_cleaned, palette='coolwarm')
plt.title("Employment Variation Rate Distribution by Housing Loan and Subscription Status")
plt.xlabel("Housing Loan")
plt.ylabel("Employment Variation Rate")
plt.legend(title="Subscribed")
plt.show()
```
![download (5)](https://github.com/user-attachments/assets/3a907645-f525-4c43-8fd8-e5e21a815354)

```python
# Visualization 7: Sunburst Chart of Marital Status, Education, and Subscription
import plotly.express as px

sunburst_data = df_cleaned.groupby(['marital', 'education', 'y']).size().reset_index(name='count')
fig = px.sunburst(sunburst_data, path=['marital', 'education', 'y'], values='count', color='count',
                  color_continuous_scale='Viridis', title="Sunburst Chart of Marital Status, Education, and Subscription")
fig.show()
```
<img width="1638" alt="Screenshot 2025-01-20 at 4 07 26 PM" src="https://github.com/user-attachments/assets/1dd27f7b-b56c-4502-9936-aa9f7986f9be" />

```python
# Visualization 8: Correlation Heatmap of Numerical Variables
plt.figure(figsize=(12, 8))
# Select only numerical features for correlation calculation
numerical_features = df_cleaned.select_dtypes(include=np.number)
correlation = numerical_features.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Analysis of the correlation heatmap
correlation_analysis = correlation.unstack().sort_values(ascending=False).drop_duplicates()
correlation_analysis
```
![download (6)](https://github.com/user-attachments/assets/6cdd1bcb-e130-486d-a68e-821115a947a1)

#### Correlation Heatmap Analysis:

The correlation heatmap highlights the relationships among numerical variables in the dataset. Key observations include:

#### Strong Positive Correlations:
* emp_var_rate and euribor3m (0.97): Indicates a strong linear relationship, suggesting economic trends are closely linked.
* nr_employed and euribor3m (0.95): Suggests that employment levels are heavily tied to interest rates.
* nr_employed and emp_var_rate (0.91): Confirms that employment variation aligns closely with overall employment numbers.
  
#### Moderate Positive Correlations:
* cons_price_idx with emp_var_rate (0.78) and euribor3m (0.69): Indicates the price index partially tracks broader economic indicators.
* pdays with nr_employed (0.37): Suggests a weak association between previous contact timing and employment levels.
  
#### Weak or Insignificant Correlations:
* Variables like age and campaign show negligible relationships with other numerical features, indicating limited linear dependencies.

#### Negative Correlations:
* previous and pdays (-0.59): Suggests that fewer previous contacts are associated with longer durations since the last contact.
* nr_employed and previous (-0.50): Indicates that employment levels inversely relate to prior campaign engagements.

#### Implications:
* Features like euribor3m, emp_var_rate, and nr_employed are critical indicators of economic trends and likely influence the target variable (y).
* Variables with weak correlations, such as campaign, may require deeper non-linear analysis or be deprioritized in feature importance.

### Features Engineering

```python
# Feature Engineering on Cleaned Data

# 1. Encoding Categorical Variables
# Convert categorical variables to one-hot encoded variables
df_encoded = pd.get_dummies(df_cleaned, columns=['job', 'marital', 'education', 'default', 'housing', 'loan',
                                                 'contact', 'month', 'day_of_week', 'poutcome'], drop_first=True)

# 2. Create Interaction Features
# Interaction between `campaign` and `pdays`
df_encoded['campaign_pdays_interaction'] = df_encoded['campaign'] * df_encoded['pdays']

# Interaction between `nr_employed` and `emp_var_rate`
df_encoded['employment_trend'] = df_encoded['nr_employed'] * df_encoded['emp_var_rate']

# 3. Binning Continuous Variables
# Age bins
df_encoded['age_group'] = pd.cut(df_cleaned['age'], bins=[0, 25, 35, 50, 65, 100],
                                 labels=['<25', '25-35', '36-50', '51-65', '65+'])

# Pdays bins
df_encoded['pdays_group'] = pd.cut(df_cleaned['pdays'], bins=[-1, 0, 5, 10, 999],
                                   labels=['Not Contacted', '0-5 Days', '6-10 Days', 'More than 10 Days'])

# ----> Include the new created columns in the one-hot encoding <----
df_encoded = pd.get_dummies(df_encoded, columns=['age_group', 'pdays_group'], drop_first=True)

# 4. Creating Aggregated Features
# Count of previous successful contacts
df_encoded['prev_success_contacts'] = (df_cleaned['poutcome'] == 'success').astype(int) * df_cleaned['previous']

# 5. Normalize/Scale Numerical Features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical_cols = ['age', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx',
                  'euribor3m', 'nr_employed']
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# Display the engineered dataframe
df_encoded.head()
```
#### Key Transformations
#### Encoding Categorical Variables:
* One-hot encoded variables for features like job, marital, education, etc.
#### Interaction Features:
* Created interaction terms like campaign_pdays_interaction and employment_trend to capture relationships between variables.
#### Binning Continuous Variables:
* Grouped age and pdays into meaningful categories (e.g., age groups and contact timing).
#### Aggregated Features:
* Counted previous successful contacts using the poutcome feature.
#### Normalization:
* Scaled numerical columns to a 0–1 range using MinMaxScaler.

### Data Modeling - Logistic Regression Model

```python
# Logistic Regression Model on the Updated Encoded Data

from sklearn.preprocessing import StandardScaler

# Define X and y for modeling
y = df_encoded['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Binary encoding for target variable
X = df_encoded.drop(columns=['y'])  # Exclude the target column

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Now X is defined before being used.

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Logistic Regression model
logistic_model = LogisticRegression(solver='liblinear', max_iter=2000, random_state=42)

# Train the model
logistic_model.fit(X_train, y_train)

# Predict on the test set
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate Logistic Regression Model
logistic_results = {
    'Accuracy': accuracy_score(y_test, y_pred_logistic),
    'Precision': precision_score(y_test, y_pred_logistic),
    'Recall': recall_score(y_test, y_pred_logistic),
    'F1 Score': f1_score(y_test, y_pred_logistic),
    'Classification Report': classification_report(y_test, y_pred_logistic)
}

# Display the evaluation metrics for Logistic Regression
logistic_results_df = pd.DataFrame([logistic_results]).T

print("Logistic Regression Model Results")
display(logistic_results_df)
```
<img width="345" alt="Screenshot 2025-01-20 at 4 45 52 PM" src="https://github.com/user-attachments/assets/c014a9cf-6382-46b1-9865-ab2d860a0dde" />

#### Initial Logistic Regression Results Analysis
#### Key Metrics:
#### Accuracy:
* The initial model achieved an accuracy of approximately 89.83%. While high, this metric alone can be misleading in an imbalanced dataset because it favors the majority class (no).

#### Precision:
* The precision score was 68.54%. This indicates that when the model predicted a positive outcome (yes), it was correct 68.54% of the time.

#### Recall:
* Recall was only 17.57%, showing that the model identified only a small fraction of the actual positive cases (yes). This suggests the model heavily favored the majority class, failing to capture most subscribing clients.

#### F1 Score:
* The F1 Score was 27.97%, reflecting the imbalance between Precision and Recall. This low score highlights that the model struggled to balance the trade-off between avoiding false positives and capturing true positives.

### Logistic Regression Model Refinements

```python
# Addressing ConvergenceWarning by increasing max_iter and refining solver
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Adjust parameters and increase max_iter
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']}
logistic_model_adjusted = LogisticRegression(class_weight='balanced', max_iter=5000, random_state=42)

# Perform GridSearchCV with updated parameters
grid_search_adjusted = GridSearchCV(logistic_model_adjusted, param_grid, cv=5, scoring='f1')
grid_search_adjusted.fit(X_train, y_train)

# Best parameters from the adjusted GridSearchCV
best_params_adjusted = grid_search_adjusted.best_params_

# Train the model with best parameters
adjusted_logistic_model = LogisticRegression(**best_params_adjusted, class_weight='balanced', max_iter=5000, random_state=42)
adjusted_logistic_model.fit(X_train, y_train)

# Predict on the test set
y_pred_adjusted = adjusted_logistic_model.predict(X_test)

# Evaluate the adjusted model
adjusted_results = {
    'Best Parameters': str(best_params_adjusted),
    'Accuracy': accuracy_score(y_test, y_pred_adjusted),
    'Precision': precision_score(y_test, y_pred_adjusted),
    'Recall': recall_score(y_test, y_pred_adjusted),
    'F1 Score': f1_score(y_test, y_pred_adjusted),
    'Classification Report': classification_report(y_test, y_pred_adjusted)
}

# Display the results of the adjusted model
adjusted_results_df = pd.DataFrame.from_dict(adjusted_results, orient='index', columns=['Value'])
print("\nAdjusted Logistic Regression Model Results:")
print(adjusted_results_df)
```
<img width="710" alt="Screenshot 2025-01-20 at 4 48 26 PM" src="https://github.com/user-attachments/assets/76eccaa2-96dd-47d7-baee-f5fa9f83ff20" />

The adjusted Logistic Regression model has been successfully trained and evaluated. The updated results are as follows:

* Best Parameters: {'C': 0.1, 'solver': 'liblinear'}

* Accuracy: 80.68%

* Precision: 31.93%

* Recall: 63.50%

* F1 Score: 42.50%

These improvements demonstrate better balance between Precision and Recall for the minority class (yes), as indicated by the higher Recall and F1 Score compared to previous evaluations.​

### Classification Models Comparison

```python
# Importing necessary libraries for additional models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

# Initialize models with default settings
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, solver='liblinear', random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

# Dictionary to store results
model_performance = []

# Loop through each model, fit, and evaluate
for model_name, model in models.items():
    # Record start time
    start_time = time.time()

    # Train the model
    model.fit(X_train, y_train)

    # Record end time
    train_time = time.time() - start_time

    # Evaluate the model on train and test sets
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    # Append results to the list
    model_performance.append({
        'Model': model_name,
        'Train Time (s)': round(train_time, 4),
        'Train Accuracy': round(train_accuracy, 4),
        'Test Accuracy': round(test_accuracy, 4)
    })

# Convert results to DataFrame
performance_df = pd.DataFrame(model_performance)

# Display the performance comparison
print("Model Performance Comparison:")
display(performance_df)
```
<img width="571" alt="Screenshot 2025-01-20 at 4 51 18 PM" src="https://github.com/user-attachments/assets/ed9b10c3-7cc7-4d4f-9954-b8eac41af368" />

### Model Improvements

```python
# Exploring feature importance using a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Train a Decision Tree Classifier for feature importance analysis
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Extract feature importances
feature_importances = tree_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the top features
top_features_df = importance_df.head(10)
print("Top 10 Feature Importances:")
display(top_features_df)

# Visualization of feature importances
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(top_features_df['Feature'], top_features_df['Importance'], color='skyblue')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()
```
![download (7)](https://github.com/user-attachments/assets/0e160f2f-2266-41f6-b641-2fc8778dab1d)

```python
# Refining the dataset to include only the top features
top_features = [
    'age', 'nr_employed', 'euribor3m', 'campaign',
    'campaign_pdays_interaction', 'housing_yes',
    'pdays', 'loan_yes', 'cons_conf_idx',
    'education_university.degree'
]

# Create a refined dataset
X_refined = X[top_features]  # Using only the top features for X
y_refined = y  # Target variable remains the same

# Train/Test split with refined features
X_train_refined, X_test_refined, y_train_refined, y_test_refined = train_test_split(X_refined, y_refined, test_size=0.3, random_state=42)

# Train a Logistic Regression model on the refined dataset
logistic_model_refined = LogisticRegression(max_iter=5000,solver='liblinear', random_state=42)
logistic_model_refined.fit(X_train_refined, y_train_refined)

# Predict on the test set
y_pred_refined = logistic_model_refined.predict(X_test_refined)

# Evaluate the refined model
refined_results = {
    'Accuracy': accuracy_score(y_test_refined, y_pred_refined),
    'Precision': precision_score(y_test_refined, y_pred_refined),
    'Recall': recall_score(y_test_refined, y_pred_refined),
    'F1 Score': f1_score(y_test_refined, y_pred_refined),
    'Classification Report': classification_report(y_test_refined, y_pred_refined, output_dict=False)
}

# Display the refined model results
refined_results_df = pd.DataFrame.from_dict(refined_results, orient='index', columns=['Value'])
print("\nRefined Logistic Regression Model Results:")
display(refined_results_df)
```
<img width="383" alt="Screenshot 2025-01-20 at 4 55 22 PM" src="https://github.com/user-attachments/assets/2788593f-3093-4b4b-b786-862b72b2ab9f" />

### Improvement in Precision:
* The refined model achieves a higher precision of 70.14%, indicating fewer false positives.
### Slight Increase in Accuracy:
* Accuracy improves to 89.95%, slightly higher than the initial model, despite using fewer features.
### Recall and F1 Score:
* Recall remains low at 18.43%, meaning the model still misses many true positives.
* The F1 Score improves to 29.19%, reflecting a better balance between Precision and Recall.

### Use Rando Forest Model with SMOTE to address class imbalance and improve Recall

```python
# Applying SMOTE to address class imbalance and improve Recall
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_refined, y_refined)

# Train/Test split on the resampled dataset
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

# Train a Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
random_forest.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred_rf = random_forest.predict(X_test_resampled)

# Evaluate the Random Forest model
rf_results = {
    'Accuracy': accuracy_score(y_test_resampled, y_pred_rf),
    'Precision': precision_score(y_test_resampled, y_pred_rf),
    'Recall': recall_score(y_test_resampled, y_pred_rf),
    'F1 Score': f1_score(y_test_resampled, y_pred_rf),
    'AUC-ROC': roc_auc_score(y_test_resampled, random_forest.predict_proba(X_test_resampled)[:, 1]),
    'Classification Report': classification_report(y_test_resampled, y_pred_rf, output_dict=False)
}

# Display the Random Forest model results
rf_results_df = pd.DataFrame.from_dict(rf_results, orient='index', columns=['Value'])
print("Random Forest Model Results:")
display(rf_results_df)
```
<img width="357" alt="Screenshot 2025-01-20 at 4 58 36 PM" src="https://github.com/user-attachments/assets/45fcf658-bea0-4460-a3c9-d16d97bf91f6" />

#### Observations

#### High Precision:
* Precision of 87.87% indicates the model effectively avoids false positives, crucial for accurately identifying subscribing clients.
#### Improved Recall:
* Recall of 72.32% demonstrates the model's capability to correctly identify the majority of positive cases (yes).
#### Balanced F1 Score:
* The F1 Score of 79.34% reflects a strong balance between Precision and Recall, making the model reliable for practical use.
#### AUC-ROC:
* AUC-ROC of 90.05% indicates excellent discrimination between classes, showcasing the model's robust predictive power.

### Summary
* The Random Forest model outperforms Logistic Regression and other models in terms of Recall and AUC-ROC.
* The ensemble approach effectively handles feature interactions and class imbalance, leading to improved performance.

#### Analyzing feature importance using the Random Forest model
```python
# Analyzing feature importance using the Random Forest model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Assuming X_train and y_train are already defined and include the refined features
# Recreate and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Extract feature importances
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the top features
top_feature_df = importance_df.head(10)
print("Top 10 Feature Importances from Random Forest:")
display(top_feature_df)

# Visualize feature importances
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(top_feature_df['Feature'], top_feature_df['Importance'], color='skyblue')
plt.title('Top 10 Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()
```
![download (8)](https://github.com/user-attachments/assets/102daae7-fe28-41ab-8658-0ec08e8208cc)

### Insights and Implications:
#### Economic Indicators:
* nr_employed and euribor3m are highly influential, reflecting macroeconomic conditions.
* These features indicate the importance of economic stability in influencing client behavior.

#### Campaign Metrics:
* campaign_pdays_interaction and pdays demonstrate the significance of the timing and frequency of client contacts.
#### Client Behavior and History:
* Features like poutcome_success and prev_success_contacts highlight the importance of past interactions.
#### Demographics:
* age plays a moderate role, likely reflecting the correlation between age and financial behavior.

### Recommendations:
#### Focus Marketing Efforts:
* Leverage insights from top features like nr_employed and euribor3m to time campaigns effectively.
* Tailor outreach to clients with favorable poutcome_success and prev_success_contacts histories.

#### Feature Refinement:
* Engineer more interactions among top features, such as euribor3m x emp_var_rate.
#### Visualization and Strategy:
* Create segmented marketing strategies based on these key features.
