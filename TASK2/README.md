# 📰 Fake News Detection Using Passive-Aggressive Classifier

This project is designed to detect **fake news articles** using Natural Language Processing and a machine learning model (Passive-Aggressive Classifier). It uses **TF-IDF vectorization** to process text and evaluates the model using various metrics and visualizations.

---

## 📁 Dataset

We use two datasets:

- `Fake.csv` – Fake news articles (labeled **0**)
- `True.csv` – Real news articles (labeled **1**)

These are combined into one dataframe for training.

```python
import pandas as pd

fake_df = pd.read_csv('dataset/Fake.csv')
true_df = pd.read_csv('dataset/True.csv')
fake_df['label'] = 0
true_df['label'] = 1
df = pd.concat([true_df, fake_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
df = df.dropna(subset=['text'])
```

---

## 📊 Class Distribution

We visualize the number of fake and real articles.

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label', palette='pastel')
plt.xticks([0, 1], ['Fake', 'Real'])
plt.title("Class Distribution")
plt.xlabel("News Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()
```

**📷 Output:**

![Class Distribution](class_distribution.png)

✅ **Purpose:** Ensures that both classes (fake and real) are balanced before training.

---

## 🧠 Text Preprocessing with TF-IDF

We convert raw news text into numerical vectors using `TfidfVectorizer`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = vectorizer.fit_transform(df['text'])
```

---

## ✂️ Train/Test Split

We split the data into training and validation sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_tfidf, df['label'], test_size=0.2, random_state=42)
```

---

## 🧪 Model Training

Using the Passive-Aggressive Classifier.

```python
from sklearn.linear_model import PassiveAggressiveClassifier

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)
```

---

## 🎯 Model Evaluation

We evaluate the model using accuracy, confusion matrix, and classification report.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
conf_mat = confusion_matrix(y_val, y_pred)
report = classification_report(y_val, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_mat)
print("Classification Report:")
print(report)
```

---

## 📉 Confusion Matrix Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
```

**📷 Output:**

![Confusion Matrix](confusion_matrix.png)

✅ **Purpose:** Shows the number of correctly and incorrectly predicted samples per class.

---

## 📊 Classification Report Plot

We visualize `precision`, `recall`, and `f1-score` per class.

```python
import pandas as pd
import matplotlib.pyplot as plt

report_dict = classification_report(y_val, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().drop('accuracy')

plt.figure(figsize=(10, 6))
report_df.iloc[:-1][['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10, 6), legend=True, colormap='Set2')
plt.title("Classification Metrics per Class")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("classification_report_plot.png")
plt.show()
```

**📷 Output:**

![Classification Report](classification_report_plot.png)

✅ **Purpose:** Helps assess model performance for each label (fake and real).

---

## 💾 Save Model & Vectorizer

```python
import joblib

joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
```

✅ **Purpose:** Allows the model to be reused without retraining.

---

## 🧪 Sample Prediction

You can test the model with new input:

```python
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

text = ["Donald Trump says the elections were rigged."]
text_tfidf = vectorizer.transform(text)
print("Fake" if model.predict(text_tfidf)[0] == 0 else "Real")
```

---

## 📜 Requirements

- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib

Install via:

```bash
pip install pandas matplotlib seaborn scikit-learn joblib
```


---

## 🙋 Author

**SRIJAN PAUL**  
This project was completed as part of an **Internship/Academic Submission**.

## 📜 License

MIT License


## 📄 License

This project is licensed under the MIT License.

