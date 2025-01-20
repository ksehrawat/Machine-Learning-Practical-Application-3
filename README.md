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


