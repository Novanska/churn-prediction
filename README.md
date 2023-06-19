# Churn Model Prediction

# About Files : 

1. **Folder deployment**: contains the file needed for huggingface deployment
2. **churn-prediction.ipynb** : jupyter notebook for Exploratory Data Analysis, Data Preprocessing , Modeling
3. **churn-prediction-inference.ipynb** : Model inferencing from the best model
4. **url.txt** : link to my huggingface deployment (there's EDA and Prediction)

# Objective :
Our main objective is to predict the customer will churn or not from the business, in this modeling i try to predict the customer from the customer behaviour and then feature engineering the data 

# Conclussion for modeling : 
1. I use 4 model to predict the churn rate ( Sequential API , Functional API , Sequential Improvement , Functional Improvement)
2. Using 4 layers for based models and 3 layers for improvement models
3. After running the models and predict the models . Sequential API (Improvement) is the best model because its not over-fit, get the highest recall and lowest loss.
