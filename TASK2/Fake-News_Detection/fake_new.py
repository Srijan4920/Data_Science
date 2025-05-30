import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the data
fake_df = pd.read_csv('dataset/Fake.csv')
true_df = pd.read_csv('dataset/True.csv')

# Step 2: Add labels and merge
fake_df['label'] = 0  # Fake
true_df['label'] = 1  # Real
df = pd.concat([true_df, fake_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Optional: Clean empty rows
df = df.dropna(subset=['text'])

# Step 3: Class distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label', palette='pastel')
plt.xticks([0, 1], ['Fake', 'Real'])
plt.title("Class Distribution")
plt.xlabel("News Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

# Step 4: Split into feature and target
X = df['text']
y = df['label']

# Step 5: Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_tfidf = vectorizer.fit_transform(X)

# Step 6: Split into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 7: Train the classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Step 8: Predictions and Evaluation
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
conf_mat = confusion_matrix(y_val, y_pred)
report = classification_report(y_val, y_pred)

print(f"Accuracy Score: {acc:.4f}")
print("\nConfusion Matrix:\n", conf_mat)
print("\nClassification Report:\n", report)

# Step 9: Confusion Matrix Heatmap
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

# Step 10: Classification Report Bar Plot
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

# Step 11: Save the model and vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\nModel and vectorizer saved as 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl'")
