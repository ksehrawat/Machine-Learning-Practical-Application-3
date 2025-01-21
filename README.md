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




