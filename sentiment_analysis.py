import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ==================== 1. DOWNLOAD DATASET ====================
print("=" * 60)
print("STEP 1: Downloading Dataset from Kaggle")
print("=" * 60)
path = kagglehub.dataset_download("mehmetisik/amazon-review")
print(f"Path to dataset files: {path}")

# Load dataset
import os
files = os.listdir(path)
print(f"\nFiles available: {files}")

# Cari file CSV
csv_file = [f for f in files if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

print(f"\nDataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ==================== 2. EXPLORATORY DATA ANALYSIS ====================
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

print(f"\nDataset Info:")
print(df.info())
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nStatistical summary:\n{df.describe()}")

# ==================== 3. PREPROCESSING ====================
print("\n" + "=" * 60)
print("STEP 3: Data Preprocessing")
print("=" * 60)

# Identifikasi kolom review dan rating
# Sesuaikan nama kolom berdasarkan dataset Anda
review_col = 'reviewText' if 'reviewText' in df.columns else df.select_dtypes(include='object').columns[0]
rating_col = 'overall' if 'overall' in df.columns else 'rating' if 'rating' in df.columns else df.select_dtypes(include='number').columns[0]

print(f"Review column: {review_col}")
print(f"Rating column: {rating_col}")

# Hapus missing values
df_clean = df[[review_col, rating_col]].dropna()
print(f"\nShape after removing missing values: {df_clean.shape}")

# Buat label sentimen berdasarkan rating
# Rating 1-2: Negative (0), Rating 3: Neutral (1), Rating 4-5: Positive (2)
def create_sentiment(rating):
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df_clean['sentiment'] = df_clean[rating_col].apply(create_sentiment)
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
df_clean['sentiment_label'] = df_clean['sentiment'].map(sentiment_map)

print(f"\nSentiment distribution:\n{df_clean['sentiment_label'].value_counts()}")

# Text Preprocessing Function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

print("\nPreprocessing text... (this may take a while)")
df_clean['cleaned_text'] = df_clean[review_col].apply(preprocess_text)

# Remove empty reviews after preprocessing
df_clean = df_clean[df_clean['cleaned_text'].str.len() > 0]
print(f"Shape after preprocessing: {df_clean.shape}")

print(f"\nExample of preprocessing:")
print(f"Original: {df_clean[review_col].iloc[0][:100]}...")
print(f"Cleaned: {df_clean['cleaned_text'].iloc[0][:100]}...")

# ==================== 4. VISUALIZATION ====================
print("\n" + "=" * 60)
print("STEP 4: Data Visualization")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Sentiment Distribution
df_clean['sentiment_label'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['red', 'gray', 'green'])
axes[0, 0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Sentiment')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Rating Distribution
df_clean[rating_col].value_counts().sort_index().plot(kind='bar', ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Rating Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Rating')
axes[0, 1].set_ylabel('Count')

# Plot 3: Review Length Distribution
df_clean['review_length'] = df_clean[review_col].str.len()
axes[1, 0].hist(df_clean['review_length'], bins=50, color='purple', alpha=0.7)
axes[1, 0].set_title('Review Length Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Length')
axes[1, 0].set_ylabel('Frequency')

# Plot 4: Word Count Distribution
df_clean['word_count'] = df_clean['cleaned_text'].str.split().str.len()
axes[1, 1].hist(df_clean['word_count'], bins=50, color='orange', alpha=0.7)
axes[1, 1].set_title('Word Count Distribution (After Cleaning)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Word Count')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('sentiment_analysis_eda.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'sentiment_analysis_eda.png'")
plt.show()

# ==================== 5. FEATURE EXTRACTION ====================
print("\n" + "=" * 60)
print("STEP 5: Feature Extraction (TF-IDF)")
print("=" * 60)

# Split data
X = df_clean['cleaned_text']
y = df_clean['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF feature shape: {X_train_tfidf.shape}")

# ==================== 6. MODEL TRAINING ====================
print("\n" + "=" * 60)
print("STEP 6: Model Training")
print("=" * 60)

# Model 1: Naive Bayes
print("\nTraining Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_pred)
print(f"Naive Bayes Accuracy: {nb_acc:.4f}")

# Model 2: Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

# ==================== 7. MODEL EVALUATION ====================
print("\n" + "=" * 60)
print("STEP 7: Model Evaluation")
print("=" * 60)

# Choose best model
best_model = lr_model if lr_acc > nb_acc else nb_model
best_pred = lr_pred if lr_acc > nb_acc else nb_pred
best_name = "Logistic Regression" if lr_acc > nb_acc else "Naive Bayes"

print(f"\nBest Model: {best_name}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, best_pred, 
                          target_names=['Negative', 'Neutral', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix.png'")
plt.show()

# ==================== 8. PREDICTION FUNCTION ====================
print("\n" + "=" * 60)
print("STEP 8: Testing Prediction Function")
print("=" * 60)

def predict_sentiment(text):
    """Predict sentiment of a given text"""
    cleaned = preprocess_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = best_model.predict(vectorized)[0]
    probability = best_model.predict_proba(vectorized)[0]
    
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    
    print(f"\nOriginal text: {text}")
    print(f"Cleaned text: {cleaned}")
    print(f"Predicted sentiment: {sentiment_labels[prediction]}")
    print(f"Confidence: {probability[prediction]:.2%}")
    print(f"Probabilities: Negative={probability[0]:.2%}, Neutral={probability[1]:.2%}, Positive={probability[2]:.2%}")
    
    return sentiment_labels[prediction]

# Test examples
test_texts = [
    "This product is absolutely amazing! Best purchase ever!",
    "Terrible quality, waste of money. Very disappointed.",
    "It's okay, nothing special but does the job."
]

for text in test_texts:
    predict_sentiment(text)
    print("-" * 60)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\nFinal Results:")
print(f"- Best Model: {best_name}")
print(f"- Accuracy: {max(nb_acc, lr_acc):.4f}")
print(f"- Total samples analyzed: {len(df_clean)}")
print(f"- Training samples: {len(X_train)}")
print(f"- Test samples: {len(X_test)}")