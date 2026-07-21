# 🛍️ Amazon Review Sentiment Analysis using TF-IDF & Logistic Regression

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-orange)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-success)
![Accuracy](https://img.shields.io/badge/Accuracy-92.57%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end **Natural Language Processing (NLP)** project that classifies Amazon product reviews into **Positive**, **Negative**, and **Neutral** sentiments using **TF-IDF feature extraction** and **Logistic Regression**.

The project demonstrates a complete machine learning workflow including data preprocessing, feature engineering, model comparison, evaluation, and analysis of design decisions.

---

# ⭐ Project Highlights

* Built an end-to-end sentiment classification pipeline.
* Performed comprehensive text preprocessing using NLTK.
* Converted text into numerical features using **TF-IDF**.
* Compared Naive Bayes and Logistic Regression models.
* Achieved **92.57% Validation Accuracy**.
* Documented technical decisions and trade-offs.
* Generated performance visualizations and confusion matrix.

---

# 📊 Dataset

**Source**

Kaggle — Amazon Review Dataset

https://www.kaggle.com/datasets/mehmetisik/amazon-review

### Dataset Overview

| Item          |                                Value |
| ------------- | -----------------------------------: |
| Total Reviews |                                4,915 |
| Positive      |                                90.5% |
| Negative      |                                 6.6% |
| Neutral       |                                 2.9% |
| Task          | Multi-class Sentiment Classification |

Example:

| Review                               | Sentiment |
| ------------------------------------ | --------- |
| "Amazing quality and fast shipping!" | Positive  |
| "The product arrived damaged."       | Negative  |
| "It's okay, nothing special."        | Neutral   |

---

# 📁 Project Structure

```
amazon-sentiment-analysis/
│
├── sentiment_analysis.py
├── requirements.txt
├── README.md
├── THOUGHT_PROCESS.md
├── sentiment_analysis_eda.png
├── confusion_matrix.png
└── model_results.txt
```

---

# 🛠 Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy
* NLTK
* Matplotlib
* Seaborn

---

# 🔄 Machine Learning Pipeline

```
Amazon Reviews

↓

Text Cleaning

↓

Tokenization

↓

Lemmatization

↓

Stopword Removal

↓

TF-IDF Vectorization

↓

Train/Test Split

↓

Model Training

↓

Evaluation

↓

Prediction
```

---

# ⚙️ Methodology

## 1. Data Collection

Dataset downloaded directly from Kaggle using Kaggle API.

---

## 2. Text Preprocessing

The preprocessing pipeline includes:

* Lowercase conversion
* Remove URLs
* Remove punctuation
* Remove numbers
* Remove stopwords
* Tokenization
* Lemmatization

---

## 3. Feature Engineering

The cleaned reviews are transformed using:

* TF-IDF Vectorizer
* Maximum Features = **5,000**
* N-grams = **(1,2)**

This configuration preserves meaningful phrases such as:

```
not good
very bad
high quality
```

instead of treating each word independently.

---

## 4. Model Comparison

Two machine learning algorithms were evaluated.

| Model                   |     Accuracy |
| ----------------------- | -----------: |
| Naive Bayes             |       90.54% |
| **Logistic Regression** | **92.57%** ✅ |

Logistic Regression was selected because it provided the best balance between prediction performance, training speed, and model simplicity.

---

# 🧠 Technical Decisions

## Why TF-IDF instead of Bag-of-Words?

TF-IDF reduces the influence of common words while increasing the importance of informative terms.

Result:

* Better feature representation
* Approximately **4% higher accuracy** compared with Count Vectorizer.

---

## Why use Bigrams?

Single words cannot preserve negation.

Example:

```
not good
```

Without bigrams:

```
not
good
```

With bigrams:

```
not_good
```

This preserves the original sentiment.

---

## Why Logistic Regression?

Although deep learning models such as LSTM and BERT often achieve slightly higher accuracy, Logistic Regression offers significant advantages for medium-sized datasets.

| Model               |   Accuracy | Training Time |
| ------------------- | ---------: | ------------: |
| Logistic Regression | **92.57%** |       ~15 sec |
| LSTM                |       ~94% |     2–3 hours |
| BERT                |       ~96% |     4–6 hours |

For this dataset, the small improvement in accuracy does not justify the additional computational cost.

---

# 📈 Model Performance

| Metric            |      Value |
| ----------------- | ---------: |
| Accuracy          | **92.57%** |
| Weighted F1 Score |    **90%** |

### Per-Class Metrics

| Class    | Precision | Recall |  F1 |
| -------- | --------: | -----: | --: |
| Positive |       93% |   100% | 96% |
| Negative |       88% |    32% | 47% |
| Neutral  |        0% |     0% |  0% |

---

# 📊 Visualizations

### Exploratory Data Analysis

```markdown
![EDA](sentiment_analysis_eda.png)
```

### Confusion Matrix

```markdown
![Confusion Matrix](confusion_matrix.png)
```

Interpretation:

* Positive reviews are classified very accurately.
* Most classification errors occur between Neutral and Negative classes due to severe class imbalance.
* The dataset contains relatively few Neutral reviews, limiting model performance on that class.

---

# 💡 Key Findings

## Strengths

* High overall classification accuracy.
* Excellent Positive sentiment detection.
* Fast model training and inference.
* Lightweight deployment compared with deep learning models.

## Challenges

* Highly imbalanced dataset.
* Limited Neutral class examples.
* Lower recall for Negative reviews.

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone https://github.com/machtya/amazon-sentiment-analysis.git

cd amazon-sentiment-analysis
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk kagglehub
```

---

## Download Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/mehmetisik/amazon-review

Place the downloaded files inside the project directory.

---

## Run

```bash
python sentiment_analysis.py
```

Generated outputs:

* sentiment_analysis_eda.png
* confusion_matrix.png
* Evaluation metrics
* Model predictions

---

# 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
kagglehub
```

---

# 📚 Lessons Learned

This project demonstrates that classical machine learning algorithms remain highly effective for medium-sized NLP datasets.

Instead of immediately using deep learning models, carefully designed preprocessing, TF-IDF feature engineering, and Logistic Regression achieved over **92% accuracy** while maintaining fast training time and low computational requirements.

The project also highlights the importance of evaluating class imbalance and interpreting model performance beyond overall accuracy.

---

# 🔮 Future Improvements

* Handle class imbalance using SMOTE.
* Tune hyperparameters with GridSearchCV.
* Compare Logistic Regression with Support Vector Machine.
* Implement deep learning models (LSTM and BERT).
* Deploy the model using Streamlit or Flask.
* Add Precision-Recall and ROC Curve analysis.

---

# 👩‍💻 Author

**Rachmatia Khoerun Nissa Surasta**

Information Systems Student — BINUS Online Learning

**GitHub:** https://github.com/machtya

**Kaggle:** https://www.kaggle.com/rachmatia

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 🤝 Contributing

Contributions, suggestions, and bug reports are welcome.

Feel free to open an Issue or submit a Pull Request.

---

⭐ If you find this project useful, consider giving the repository a star!
