# **Repository Overview**

This repository contains the project files for an extensive analysis of COVID-19's impact on hospital capacities and staffing shortages across various states. The analysis was performed using Python, with data visualizations created in matplotlib and seaborn, and advanced statistical methods applied for deeper insights.

## **Files Included**

- COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW_.csv - Raw dataset containing detailed information on hospital capacities and staffing issues over time across the U.S. states. Due to the size of the raw data file, it is not hosted on GitHub but can be accessed directly from the HealthData.gov portal through this link. https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh/data
- Analysis Scripts and Notebooks - Python script containing the code used for data cleaning, analysis, visualization, and statistical modeling.

## **Project Description**
### **Objective**

This project aims to understand the temporal and regional variations in hospital capacities and staffing shortages during the COVID-19 pandemic across the United States. It focuses on key metrics such as inpatient beds used for COVID-19, staffing shortages, and overall hospital capacity utilization.

### **Data Analysis**

- **Data Cleaning and Preparation:** The dataset was cleaned to handle missing values, correct data types, and reduce dimensionality for focused analysis.
- **Exploratory Data Analysis (EDA):** Initial exploration to understand the structure, trends, and patterns in the data.
- **Feature Engineering:** Critical features were selected to focus on the aspects most relevant to hospital capacity and staffing shortages.
- **Visual Analysis:** Time series plots and bar charts were used to visualize trends and compare data across different states and over time.
- **Statistical Analysis:** Applied time series forecasting models like SARIMAX to predict future hospital capacity needs, and machine learning models like RandomForest and Logistic Regression for predicting staffing shortages.

### **Insights and Outputs**

- Visualizations showing the trends of hospital usage and staffing shortages over time.
- Predictive models providing insights into future trends in hospital capacity needs and staffing issues.
- Comparative analysis across states to identify regions with the most significant challenges.

## **Installation**

To interact with the analysis scripts and notebooks:
- Clone this repository to your local machine.
- Ensure Python is installed on your computer.
- Install necessary Python libraries via pip:

  pip install pandas numpy matplotlib seaborn statsmodels scikit-learn

Navigate to the folder containing the downloaded scripts and notebooks.
Run Jupyter Notebook or JupyterLab:

    jupyter notebook

## **Usage**

- Data File: Due to its size, the raw data file is not included in the repository. It can be downloaded from HealthData.gov.
- Python Notebooks: Open and run the notebooks to reproduce the analysis and view the visualizations.
