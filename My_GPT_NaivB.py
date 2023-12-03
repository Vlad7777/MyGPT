#My_GPT_NaivB


"""
Naive Bayes is a probabilistic machine learning algorithm based on Bayes' theorem. It is widely used for classification problems, particularly in natural language processing and spam filtering. Python provides several libraries, including scikit-learn, for implementing Naive Bayes classifiers. Below is an example using scikit-learn's MultinomialNB for text classification:

python
Copy code
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample data
corpus = [
    ('This is a positive document', 'positive'),
    ('This is a negative document', 'negative'),
    ('I feel great about it', 'positive'),
    ('I am not happy about it', 'negative'),
]

# Split the data into features (X) and labels (y)
X, y = zip(*corpus)

# Convert text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', report)

"""

In this example:

The CountVectorizer is used to convert the text data into numerical features.
The data is split into training and testing sets using train_test_split.
The MultinomialNB classifier is trained on the training set.
Predictions are made on the test set.
Performance is evaluated using accuracy and a classification report.
This is a basic example, and scikit-learn provides other variants of Naive Bayes classifiers, such as GaussianNB for continuous data and BernoulliNB for binary data. The choice of the Naive Bayes variant depends on the nature of your data.

Adjust the code and data based on your specific use case and dataset.


"""


