#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


inpPath = "C:/CarolineZiegler/Studium_DCU/8. Semester/Business Analytics Portfolio/Portfolio/03_Healthcare/"
COVDf = pd.read_csv(inpPath + "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW_.csv", delimiter =  ",", header = 0)
COVDf


# In[3]:


#understanding the dataset structure and input
COVDf.describe()


# In[4]:


COVDf.info()


# In[5]:


COVDf.isnull().sum()


# In[6]:


#convert 'date' column to datetime format
COVDf['date'] = pd.to_datetime(COVDf['date'], format='%Y/%m/%d')


# In[7]:


print(COVDf.columns)


# In[8]:


for column_name in COVDf.columns:
    print(column_name)


# In[9]:


#high dimensionality of dataset with 134 columns, therefore a feature selection for the most critical parts is necessary
#with the goal to understand hopsitality capacit needs better as well as critical staff shortages, the following columns were selected for further analysis
subset_COVDf = COVDf[['state','date', 'critical_staffing_shortage_today_yes', 'critical_staffing_shortage_today_no', 'inpatient_beds_used_covid', 'total_adult_patients_hospitalized_confirmed_covid',
    'total_pediatric_patients_hospitalized_confirmed_covid', 'inpatient_beds', 'inpatient_beds_used', 'total_staffed_adult_icu_beds', 'staffed_adult_icu_bed_occupancy']]
subset_COVDf


# In[10]:


subset_COVDf.isnull().sum()


# In[11]:


#There are missing values in several columns, as indicated by NaN entries. For example, the total_staffed_pediatric_icu_beds column has missing values in the first few rows


# In[12]:


#dropping misssing values
subset_COVDf.dropna(inplace = True)


# In[13]:


subset_COVDf


# In[14]:


subset_COVDf.isnull().sum()


# In[15]:


subset_COVDf.reset_index(inplace=True)
subset_COVDf


# In[16]:


subset_COVDf.drop('index', inplace = True, axis=1)


# In[17]:


subset_COVDf


# In[20]:


subset_COVDf['state'].unique()


# In[18]:


#data anylsis to understand the inpatient beds and critical staffing better
#plot inpatient beds used for COVID-19 over time for a sample state

sample_state = 'CA'
state_data = subset_COVDf[subset_COVDf['state'] == sample_state]
plt.figure(figsize=(12, 6))
plt.plot(state_data['date'], state_data['inpatient_beds_used_covid'], label='Inpatient Beds Used for COVID-19')
plt.title(f'Trends of Inpatient Beds Used for COVID-19 in {sample_state} Over Time')
plt.xlabel('Date')
plt.ylabel('Inpatient Beds Used for COVID-19')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()


# In[19]:


#the plot shows the trend of inpatient beds used for COVID-19 in California. It provides a visual representation of how the usage of inpatient beds for COVID-19 patients has fluctuated over time being crucial for understanding the impact of COVID-19 waves and the resulting pressure on hospital capacities.


# In[25]:


sample_state2 = 'NY'
state_data2 = subset_COVDf[subset_COVDf['state'] == sample_state2]
plt.figure(figsize=(12, 6))
plt.plot(state_data2['date'], state_data2['inpatient_beds_used_covid'], label='Inpatient Beds Used for COVID-19', c = 'green')
plt.title(f'Trends of Inpatient Beds Used for COVID-19 in {sample_state2} Over Time')
plt.xlabel('Date')
plt.ylabel('Inpatient Beds Used for COVID-19')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()


# In[23]:


#similar structure of fluctuations; however, CA did have many more inpatient beds


# In[26]:


sample_state3 = 'OK'
state_data3 = subset_COVDf[subset_COVDf['state'] == sample_state3]
plt.figure(figsize=(12, 6))
plt.plot(state_data3['date'], state_data3['inpatient_beds_used_covid'], label='Inpatient Beds Used for COVID-19', c = 'orange')
plt.title(f'Trends of Inpatient Beds Used for COVID-19 in {sample_state3} Over Time')
plt.xlabel('Date')
plt.ylabel('Inpatient Beds Used for COVID-19')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()


# In[27]:


sample_state4 = 'TX'
state_data4 = subset_COVDf[subset_COVDf['state'] == sample_state4]
plt.figure(figsize=(12, 6))
plt.plot(state_data4['date'], state_data4['inpatient_beds_used_covid'], label='Inpatient Beds Used for COVID-19', c = 'red')
plt.title(f'Trends of Inpatient Beds Used for COVID-19 in {sample_state4} Over Time')
plt.xlabel('Date')
plt.ylabel('Inpatient Beds Used for COVID-19')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()


# In[28]:


sample_state5 = 'FL'
state_data5 = subset_COVDf[subset_COVDf['state'] == sample_state5]
plt.figure(figsize=(12, 6))
plt.plot(state_data5['date'], state_data5['inpatient_beds_used_covid'], label='Inpatient Beds Used for COVID-19', c = 'purple')
plt.title(f'Trends of Inpatient Beds Used for COVID-19 in {sample_state5} Over Time')
plt.xlabel('Date')
plt.ylabel('Inpatient Beds Used for COVID-19')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()


# In[30]:


import seaborn as sns


# In[31]:


#critical staffing shortage situations over time by state
states_to_compare = ['CA', 'NY', 'OK', 'TX', 'FL']
shortage_comparison_data = subset_COVDf[subset_COVDf['state'].isin(states_to_compare)]
plt.figure(figsize=(12, 6))
sns.lineplot(data=shortage_comparison_data, x='date', y='critical_staffing_shortage_today_yes', hue='state')
plt.title('Critical Staffing Shortage Situations Over Time by State')
plt.xlabel('Date')
plt.ylabel('Number of Hospitals with Critical Staffing Shortages')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[33]:


#critical staffing shortage was especially between 07/2020 and 01/2022 highlighting the temporal patterns of staffing shortages and differences between these states


# In[34]:


#understanding the hospital capacity utilization on a national level over time better (more holistic analysis than only some states)
#calculate the national average of bed utilization over time
national_avg_utilization = subset_COVDf.groupby('date')['inpatient_beds_used_covid'].mean().reset_index()
national_avg_utilization


# In[36]:


plt.figure(figsize=(14, 7))
sns.lineplot(data=national_avg_utilization, x='date', y='inpatient_beds_used_covid')
plt.title('Average Inpatient Beds Used for COVID-19 Over Time (USA National Level)')
plt.xlabel('Date')
plt.ylabel('Average Inpatient Beds Used')
plt.show()


# In[37]:


#staffing shortages over time on a national level
#calculate the national average of staffing shortages over time
national_avg_staffing_shortage = subset_COVDf.groupby('date')['critical_staffing_shortage_today_yes'].mean().reset_index()
national_avg_staffing_shortage


# In[39]:


plt.figure(figsize=(14, 7))
sns.lineplot(data=national_avg_staffing_shortage, x='date', y='critical_staffing_shortage_today_yes', c = "green")
plt.title('Average Reported Critical Staffing Shortage Over Time (USA National Level)')
plt.xlabel('Date')
plt.ylabel('Average Critical Staffing Shortage')
plt.show()


# In[41]:


#comparison analysis of the states
#compare the latest date in the dataset
latest_date = subset_COVDf['date'].max()
state_comparison = subset_COVDf[subset_COVDf['date'] == latest_date]


# In[42]:


plt.figure(figsize=(14, 7))
sns.barplot(data=state_comparison, x='state', y='inpatient_beds_used_covid')
plt.title('Inpatient Beds Used for COVID-19 by State')
plt.xlabel('State')
plt.ylabel('Inpatient Beds Used for COVID-19')
plt.xticks(rotation=90)
plt.show()


# In[44]:


#compare the earliest date in the dataset
earliest_date = subset_COVDf['date'].min()
state_comparison2 = subset_COVDf[subset_COVDf['date'] == earliest_date]


# In[45]:


plt.figure(figsize=(14, 7))
sns.barplot(data=state_comparison2, x='state', y='inpatient_beds_used_covid')
plt.title('Inpatient Beds Used for COVID-19 by State')
plt.xlabel('State')
plt.ylabel('Inpatient Beds Used for COVID-19')
plt.xticks(rotation=90)
plt.show()


# In[46]:


#the state NV was the first to register inpatient beds


# In[47]:


#with these insights, a more detailed comparative analysis of hospital capacities over time and between states can be done through predictive analysis, focusing on identifying key patterns, seasonalities, and anomalies


# In[48]:


#time series forecasting for future hospital capacity needs
#model and forecast future hospital capacity needs, employing time series analysis techniques
#considering the seasonal patterns and potential for recurring waves of infections, models like ARIMA or SARIMA will be applied and tested 


# In[51]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


# In[52]:


#data preparation
ts_data = subset_COVDf[subset_COVDf['state'] == 'CA'][['date', 'inpatient_beds_used_covid']].copy()
ts_data.set_index('date', inplace=True)
ts_data = ts_data.asfreq('D').fillna(method='ffill')


# In[53]:


#check for stationarity and prepare data
if adfuller(ts_data['inpatient_beds_used_covid'])[1] >= 0.05:
    ts_data_diff = ts_data.diff().dropna()
else:
    ts_data_diff = ts_data


# In[58]:


#splitting the data and training a SARIMAX model
train_data = ts_data_diff[:int(0.8 * len(ts_data_diff))]
test_data = ts_data_diff[int(0.8 * len(ts_data_diff)):]
model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)
forecast = model_fit.forecast(steps=len(test_data))

print('Model Fit:', model_fit)
print('Forecast:', forecast)


# In[59]:


#visualising the forecast vs actuals
forecast_series = pd.Series(forecast, index=test_data.index)
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['inpatient_beds_used_covid'], label='Training Data')
plt.plot(test_data.index, test_data['inpatient_beds_used_covid'], label='Actual Value')
plt.plot(forecast_series.index, forecast_series, label='Forecasted Value')
plt.title('Forecast vs Actuals for Inpatient Beds Used for COVID-19 in CA')
plt.xlabel('Date')
plt.ylabel('Inpatient Beds Used for COVID-19')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[60]:


#the model predicted a continuous decrease; however, the actual numbers increased in the trend


# In[63]:


#classification for predicting critical staffing shortages
#aim to predict whether a hospital will face a critical staffing shortage
#RandomForestClassifier was chosen for its robustness to overfitting, ability to handle non-linear data, and ease of use without the need for feature scaling


# In[62]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[64]:


classification_features = [
    'inpatient_beds_used_covid', 'total_adult_patients_hospitalized_confirmed_covid',
    'total_pediatric_patients_hospitalized_confirmed_covid', 'inpatient_beds',
    'inpatient_beds_used', 'total_staffed_adult_icu_beds', 'staffed_adult_icu_bed_occupancy'
]
X = subset_COVDf[classification_features]
y = subset_COVDf['critical_staffing_shortage_today_yes'] > 0  # Binary classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[65]:


# RandomForestClassifier model training and evaluation
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[66]:


#the RandomForestClassifier model for predicting critical staffing shortages yielded an accuracy of approximately 95.46%
#precision: The model's precision is high across both classes, with 91% for hospitals not facing shortages and 97% for those facing shortages. This indicates a high likelihood that the model's predictions are correct when it predicts a specific class.
#recall: The recall is also strong, at 89% for hospitals not facing shortages and 97% for those facing shortages, suggesting the model is good at identifying actual instances of each class.
#F1-Score: The F1-scores, which balance precision and recall, are 90% and 97% for the two classes, respectively, indicating robust overall performance.


# In[67]:


#RandomForestClassifier seems to work quite well
#LogisticRegression will be performed to have a second model to compare the results which will allow for recommending which model to be best used on the data
#Logistic Regression is useful for binary classification problems and understanding the influence of several independent variables on a single outcome variable which is why it was chosen for this dataset


# In[68]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[69]:


# Train the Logistic Regression model
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train, y_train)


# In[70]:


# Predict on the test set
y_pred_lr = lr_classifier.predict(X_test)
y_pred_lr


# In[72]:


# Evaluate the Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr)
print('Logistic Regression Accuracy:', accuracy_lr)
print('\nLogistic Regression Classification Report:\n', report_lr)


# In[73]:


#the LogisticRegression model for predicting critical staffing shortages yielded an accuracy of approximately 77.04%
#precision: the model's precision is medium across both classes, with 60% for hospitals not facing shortages and 78% for those facing shortages. This indicates a high likelihood that the model's predictions are correct when it predicts a specific class.
#recall: the recall is quite low, at 7% for hospitals not facing shortages and quite high with 99% for those facing shortages, suggesting the model is not quite good at identifying actual instances of each class.
#F1-Score: the F1-scores, which balance precision and recall, are 13% and 87% for the two classes, respectively, indicating not a robust performance of the model


# In[74]:


#RandomForestClassifier should be the preferred model for predicting critical staff shortages being robust with a high accuracy score

