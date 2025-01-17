---

# **Customer Churn Prediction**

## **Overview**
Customer churn is a critical challenge for businesses, as retaining existing customers is often more cost-effective than acquiring new ones. This project aims to predict customer churn using machine learning techniques. The insights derived from this project can help businesses proactively address customer retention.

---

## **Project Workflow**
1. **Problem Definition**  
   - Objective: Predict whether a customer will churn based on their demographics, usage patterns, and account information.
   - Dataset: [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn).

2. **Data Cleaning**  
   - Removed irrelevant columns.
   - Handled missing values and converted categorical data into numerical format.

3. **Exploratory Data Analysis (EDA)**  
   - Visualized churn distribution and feature correlations.
   - Identified key factors influencing churn.

4. **Feature Engineering**  
   - Scaled numerical features for better model performance.
   - Encoded categorical variables.

5. **Model Building**  
   - Used a Random Forest Classifier for prediction.
   - Evaluated the model using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

6. **Deployment** (Optional)  
   - Created a simple web app using Flask to predict churn for new customers.

---

## **Technologies Used**
- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Deployment**: Flask (optional)  

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/hsrahh/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) and place it in the project folder.

4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

5. (Optional) To run the Flask app:
   ```bash
   python app.py
   ```

---

## **Usage**
- Open the Jupyter Notebook and follow the step-by-step implementation to explore the dataset, train the model, and evaluate performance.
- Use the Flask app to input customer data and predict churn.

---

## **Results**
- Achieved an accuracy of **XX%** and an ROC-AUC score of **XX**.
- Key insights:
  - Customers with shorter tenures are more likely to churn.
  - Certain services (e.g., streaming) significantly impact churn rates.

---

## **File Structure**
```
customer-churn-prediction/
├── Telco-Customer-Churn.csv   # Dataset
├── churn_prediction.ipynb     # Jupyter Notebook with code and analysis
├── app.py                     # Flask app for deployment
├── model.pkl                  # Trained model
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## **Future Work**
- Experiment with other machine learning algorithms like XGBoost or LightGBM.
- Improve feature engineering for better model performance.
- Deploy the project on cloud platforms like Heroku or AWS.

---

## **Contributing**
Contributions are welcome! Feel free to fork the repository and submit a pull request.

---
