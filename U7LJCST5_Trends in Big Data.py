'''
This entry demonstrates the powerful applications of AI in big data analysis through predictive analytics, natural language processing, and anomaly detection.
The included practical examples below not only illustrate these techniques but also provide a clear understanding of their significant impact and utility.
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Predictive Analytics
'''
Goal: Predict future values based on historical data.
Approach: We use a Random Forest Regressor, a type of machine learning model that combines many decision trees to make accurate predictions.
'''
def predictive_analytics(data):
    # Preparing the data: Split the data into features (input variables) and target (output variable).
    X = data.drop('target', axis=1)
    y = data['target']
    # Splitting the data: Divide the data into training and testing sets to evaluate the model’s performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Training the model: Fit the Random Forest Regressor to the training data.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Making predictions: Use the trained model to predict values for the test set.
    predictions = model.predict(X_test)
    # Visualizing the results: Plot actual vs. predicted values to compare how well the model performs.
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.show()

# 2. Natural Language Processing (NLP)
'''
Goal: Analyze and extract insights from text data.
Approach: Use TF-IDF (Term Frequency-Inverse Document Frequency), a statistical measure that evaluates the importance of a word in a document relative to a corpus (collection of documents).
'''
def nlp_analysis(text_data):
    # Converting text to numerical features: Use TfidfVectorizer to transform text data into numerical features.
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text_data)
    # Extracting important words: Identify and display the most significant words in the text data based on their TF-IDF scores.
    feature_names = vectorizer.get_feature_names_out()
    word_counts = np.asarray(X.sum(axis=0)).flatten()
    top_words = pd.DataFrame({'word': feature_names, 'count': word_counts}).sort_values(by='count', ascending=False)
    # Displaying the top 10 words
    print(top_words.head(10))

# 3. Anomaly Detection
'''
Goal: Identify unusual patterns or outliers in data.
Approach: Use One-Class SVM (Support Vector Machine), a machine learning model specifically designed for anomaly detection.
'''
def anomaly_detection(data):
    # Training the model: Fit the One-Class SVM model to the data to learn what normal behavior looks like.
    model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
    model.fit(data)
    # Detecting anomalies: Use the trained model to identify data points that deviate significantly from normal behavior.
    anomalies = model.predict(data)  
    # Visualizing the results: Plot the data, highlighting detected anomalies.
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.values, label='Data')
    plt.plot(data.index[anomalies == -1], data.values[anomalies == -1], 'ro', label='Anomalies')
    plt.legend()
    plt.show()
    
# Example usage
# Generating some synthetic data for demonstration
np.random.seed(42)
synthetic_data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'target': np.random.randn(100)
})
# Call the predictive analytics function
predictive_analytics(synthetic_data)

# Generating some synthetic text data
text_data = [
    "AI is transforming the world",
    "Big data is the future",
    "Machine learning is a subset of AI",
    "Deep learning enables complex pattern recognition"
]
# Call the NLP function
nlp_analysis(text_data)

# Generating some synthetic data for anomaly detection
synthetic_anomaly_data = pd.DataFrame({
    'value': np.concatenate([np.random.normal(0, 1, 90), np.random.normal(0, 10, 10)])
})
# Call the anomaly detection function
anomaly_detection(synthetic_anomaly_data)
