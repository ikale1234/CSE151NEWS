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

2. **Model**
   - XGBoost Classifier on feature engineered numerical data
   - Tuned parameters such as `n_estimators`, `learning_rate`, and `max_depth`, `lambda`, and `alpha`
   - Used Graph Search in order to optimize these parameters

3. **Evaluation Metrics**  
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
-    The XGBoost model achieves a balance between bias and variance, as evidenced by its low training error (4%) and low test error (9%). The inclusion of L1 (Lasso) and L2 (Ridge) regularization in XGBoost helps prevent overfitting by penalizing overly complex models, which keeps the variance low while maintaining predictive performance.

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
- The second model (XGBoost) demonstrates strong performance, relatively similar to the first model(Logistic Regression), although with less likelihood that it is just memorizing the data due to the engineered features as well as the reduced dimensionality. With 96% training accuracy and 91% test accuracy, we can tell that the model is able to generalize well, and it is likely not overfitting due to hyperparameter optimization.
   
- **Potential Improvements:**  
   - Hyperparameter optimization via Bayesian optimization.  
   - Use advanced NLP embeddings (e.g., TF-IDF vectors or word embeddings like Word2Vec, GloVe) to enhance text representation.  
   - Experiment with dropout or other regularization techniques to minimize overfitting further.


