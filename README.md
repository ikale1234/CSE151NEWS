### Overview
This project explores text classification for identifying fake news articles. Utilizing advanced natural language processing techniques and machine learning models, the goal is to improve prediction accuracy for imbalanced datasets.


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

2. **SMOTE for Balancing Classes**  
   - Applied SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance, ensuring equal representation of classes in the training data.

3. **Model**
   - XGBoost Classifier on feature engineered numerical data
   - Tuned parameters such as `n_estimators`, `learning_rate`, and `max_depth`.

4. **Evaluation Metrics**  
   - Used classification metrics such as accuracy, precision, recall, and F1-score to evaluate model performance on training and testing datasets.

---



### Model 1 Results

#### Fitting Graph Analysis:
- Logistic Regression fitting graph showed higher error rates for lower regularization (`C`), indicating that increasing `C` improved performance up to a certain threshold.

#### Model Performance:  
- **Training Accuracy:** 100%  
- **Testing Accuracy:** 97%



#### Analysis:  
Our first logistic regression model fared pretty well, with about 97% accuracy on the test data, although we recognize that there is still room for optimization. A possible strategy for improving this model will involve comprehensive error analysis with case examination of misclassified observations in search of a pattern in model failures. The results from these will be used to further tune the hyperparameters of the current model.

When looking at specific words that impacted the model's predictions, we extracted the 10 words that correspond to the 10 most impactful coefficients. These words are 'wednesday', 'image', 'via', 'reuters', 'didnt', 'doesnt', 'corrupt', 'lie', 'duke', and 'wire'. While some don't really show much, words like corrupt and lie, and references to duke and reuters seem to determine whether an article should be classified as fake according to this model.





### Model 2 Results

#### Fitting Graph Analysis:
- The XGBoost classifier achieved optimal performance with training error around **0.04** and testing error around **0.09**, demonstrating that the model is well-fitted with minimal overfitting.

#### Model Performance:  
- **Training Accuracy:** 96%  
- **Testing Accuracy:** 91%  
- Demonstrates robustness and good generalization. 

#### Next Models to Explore:  
- **Random Forest Classifier:**  
   To investigate ensemble methods with high interpretability. It’s less sensitive to outliers and can handle feature interactions better.
- **Deep Learning Approaches:**  
   LSTMs or transformers (e.g., BERT) to better capture the sequential structure and semantics of textual data.

---


#### Analysis:  
- Outperformed Logistic Regression with significantly better metrics. The SMOTE-balanced training data contributed to this improvement.  
- **Potential Improvements:**  
   - Hyperparameter optimization via grid search or Bayesian optimization.  
   - Use advanced NLP embeddings (e.g., TF-IDF vectors or word embeddings like Word2Vec, GloVe) to enhance text representation.  
   - Experiment with dropout or other regularization techniques to minimize overfitting further.


