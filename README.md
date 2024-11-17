# Milestone 3

---

# Text Preprocessing Pipeline

This repository contains a text preprocessing pipeline implemented using Python and the Natural Language Toolkit (NLTK). The steps included in the pipeline focus on text cleaning, tokenization, lemmatization, and removal of stopwords, among other preprocessing tasks. This pipeline is designed to prepare textual data for further analysis or machine learning tasks.

## Steps Overview

1. **Install Dependencies**: Ensure all required libraries (e.g., NLTK) are installed.
2. **Download NLTK Datasets**: Use `nltk.download('all')` to download necessary datasets, corpora, and models.
3. **Clean Text**: Remove unwanted characters and convert the text to lowercase.
4. **Tokenize Text**: Split the text into individual words using NLTK's word_tokenize function.
5. **Remove Stopwords**: Filter out common words that do not contribute to meaning in the analysis.
6. **Lemmatize Tokens**: Reduce words to their base form for consistency and better analysis.

---

## Prerequisites

Before running the code, ensure you have the following Python libraries installed:

- `nltk`
- `re` (built-in Python library)
- `string` (built-in Python library)

You can install NLTK using pip if you don’t have it already:

```bash
pip install nltk
```

## Setup and Imports

The code begins with importing necessary libraries and downloading the required NLTK datasets:

```python
import string
import nltk
nltk.download('all')
import re
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
```

- **string**: Provides a collection of string constants such as punctuation.
- **nltk**: The core library for natural language processing tasks. We use it for tasks like tokenization, lemmatization, and stopword filtering.
- **re**: Used for regular expressions and text cleaning.
- **TfidfVectorizer**: Used for TF-IDF text analysis.

The `nltk.download('all')` command downloads all NLTK datasets, corpora, and models required for preprocessing.

## Preprocessing Steps

### 1. Text Cleaning

In this step, we cleaned the input text by removing unwanted characters or formatting, such as extra spaces or punctuation:

```python
def clean_text(text):
    # Remove non-alphanumeric characters and convert text to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text
```

- **Regular Expression (`re.sub`)**: Used to remove non-alphanumeric characters, such as punctuation, and convert the text to lowercase.

### 2. Tokenization

After cleaning the text, the next step is tokenization, which involves splitting the text into individual words (tokens):

```python
tokens = word_tokenize(cleaned_text)
```

- **word_tokenize**: This NLTK function splits the text into tokens (words).

### 3. Removing Stopwords

Stopwords are common words (e.g., "the", "and", "is") that do not contribute significant meaning to text analysis. These words are removed from the text during preprocessing:

```python
stop_words = set(stopwords.words('english'))
tokens_no_stopwords = [word for word in tokens if word not in stop_words]
```

- **stopwords.words('english')**: Retrieves a list of English stopwords from the NLTK library.
- **List Comprehension**: Filters out stopwords from the list of tokens.

### 4. Lemmatization

Lemmatization is the process of reducing words to their base or root form (e.g., “running” becomes “run”). The WordNet Lemmatizer is used for this task:

```python
lemmatizer = WordNetLemmatizer()
tokens_lemmatized = [lemmatizer.lemmatize(word) for word in tokens_no_stopwords]
```

- **WordNetLemmatizer()**: The lemmatizer provided by NLTK, which uses the WordNet lexical database to find the base form of words.

### 5. Text Analysis

Term Frequency-Inverse Document Frequency (TF-IDF) is a Natural Language Processing (NLP) used to measuring the importance of a word within the collection of words by considering how often the word appears within the document. The Tfidf Vectorizer is used for this process:

```python
tfidf = TfidfVectorizer()
result = tfidf.fit_transform(corpus)
```

### 6. Output

Finally, after performing the preprocessing steps, you have a list of cleaned, tokenized, and lemmatized words that can be used for further analysis or machine learning tasks.

```python
print(tokens_lemmatized)
```

---

## Example

```python
text = "This is an example sentence, which needs preprocessing!"
cleaned_text = clean_text(text)
tokens = word_tokenize(cleaned_text)
tokens_no_stopwords = [word for word in tokens if word not in stop_words]
tokens_lemmatized = [lemmatizer.lemmatize(word) for word in tokens_no_stopwords]
print(tokens_lemmatized)
```

### Output:
```
['example', 'sentence', 'need', 'preprocessing']
```

---

### Model Evaluation

The bag of words logistic regression model was able to achieve training 100% accuracy, and it seemed to generalize super well to the test data (97% accuracy). In terms of how the model fit in the fitting graph, the numbers show that it is in the ideal range for the model complexity (as it has similar training and testing accuracy). This also makes sense because the model is not super complex (vocab size of features inputted into a single logistic regression).

## Next Possible Models

In terms of our next models, we were thinking about experimenting with RNN models like LSTMs on the text, as they would be able to capture the positioning of words in our text as well as its meaning (compared to bag of words which just counts them). We want to also explore Transformer Encoder models on the text, as well as regression models on our numerical features derived from feature engineering (lexical diversity, text length, etc) to see if they are able to predict an article's validity.
