# Project Summary: Predicting Term Deposit Subscriptions Using Machine Learning

## Objective
The goal of this project was to develop machine learning models to predict whether clients of a Portuguese bank would subscribe to a term deposit based on demographic, financial, and campaign-related data. The project aimed to provide actionable insights for marketing strategies while optimizing model performance.

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
