### Model 1 Method
1. **Data Preprocessing**  
   - Applied CountVectorizer on the text data on each observation, which counts each word within the dataset vocabulary and creates a vector for the counts of each possible word. This is a bag of words vector.

2. **Model**
   - Logistic Regression model on bag of words vector
   - Tuned C parameter

3. **Evaluation Metrics**  
   - Used classification metrics such as accuracy, precision, recall, and F1-score to evaluate model performance on training and testing datasets.


### Model 2 Method
1. **Data Preprocessing**  
   - Cleaned and transformed text data into features like lexical diversity, average word lengths, sentence counts, and more.
   - Dropped unnecessary columns and separated data into features (X) and labels (y).

2. **Model**
   - XGBoost Classifier on feature engineered numerical data
   - Tuned parameters such as `n_estimators`, `learning_rate`, and `max_depth`, `lambda`, and `alpha`
   - Used Graph Search in order to optimize these parameters

3. **Evaluation Metrics**  
   - Used classification metrics such as accuracy, precision, recall, and F1-score to evaluate model performance on training and testing datasets.
