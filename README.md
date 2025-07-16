# 📧 Spam Detection using Naive Bayes

A simple and efficient **spam classifier** built using `Multinomial Naive Bayes` and `CountVectorizer`. It classifies text messages as either **spam** or **not spam** (ham).

---

## 🚀 Overview

This project uses the classic **SMS Spam Collection Dataset** to train a machine learning model for binary classification.

It supports:

- Preprocessing the dataset  
- Converting text to numerical vectors using **CountVectorizer**  
- Training a **Multinomial Naive Bayes** classifier  
- Making predictions on new messages  
- Measuring model accuracy  
- Building a **Pipeline** to simplify the workflow  

---

## 🧠 Model Architecture

- **Text Vectorization**: `CountVectorizer` is used to convert raw messages into token count matrices.
- **Classification Algorithm**: `MultinomialNB`, best suited for discrete features like word counts.

---

## 📁 Dataset

Uses the `spam.csv` dataset, which contains two main columns:
- `Category`: `spam` or `ham`
- `Message`: The actual text message

---

## 🛠️ Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- joblib (for optional model saving/loading)

---

## ⚙️ How It Works

1. Load and explore the dataset  
2. Convert the `Category` column to binary (spam = 1, ham = 0)  
3. Split the dataset into training and testing sets  
4. Vectorize the text messages  
5. Train a Naive Bayes classifier  
6. Evaluate accuracy  
7. Predict on custom messages  
8. Use `Pipeline` to streamline the entire process

---

## 🧪 Sample Predictions

```python
emails = [
    'Hey mohan, can we get together to watch football game tomorrow?',
    'Upto 20 per cent discount on parking, exclusive offer just for you. Dont miss this reward!'
]

print(clf.predict(emails))  
# Output: [0 1] => First is ham, second is spam
