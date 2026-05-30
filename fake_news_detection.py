# Fake News Detection Using Naive Bayes
# NLP Mid-Term Project - Group 7
# American International University-Bangladesh (AIUB)

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# ── Load Dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("/content/drive/MyDrive/fake_and_real_news.csv")
print(df.head())
print(df.columns)

# ── Task 1: Minimum Edit Distance ─────────────────────────────────────────────
def edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def text_similarity(str1, str2):
    distance = edit_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    similarity = 1 - (distance / max_len) if max_len > 0 else 1
    return similarity

if __name__ == "__main__":
    text1 = input("Enter first text: ")
    text2 = input("Enter second text: ")
    print(f"\nMinimum Edit Distance: {edit_distance(text1, text2)}")
    print(f"Similarity Score: {text_similarity(text1, text2):.4f}")

# ── Task 2: Preprocessing Pipeline ───────────────────────────────────────────

# 1. Word Tokenization
df['tokenized'] = df['Text'].apply(lambda x: word_tokenize(str(x)))
print(df[['Text', 'tokenized']].head())

# 2. Case Folding
df['lowercase'] = df['tokenized'].apply(lambda x: [word.lower() for word in x])
print(df['lowercase'].head())

# 3. Synonym Substitution
def get_synonym(word):
    syns = wordnet.synsets(word)
    if syns:
        return syns[0].lemmas()[0].name()
    return word

df['synonym_sub'] = df['lowercase'].apply(lambda x: [get_synonym(word) for word in x])
print(df['synonym_sub'].head())

# 4. Stemming
stemmer = PorterStemmer()
df['stemmed'] = df['synonym_sub'].apply(lambda x: [stemmer.stem(word) for word in x])
print(df['stemmed'].head())

# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
df['lemmatized'] = df['stemmed'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
print(df['lemmatized'].head())

# 6. Punctuation Removal
df['no_punct'] = df['lemmatized'].apply(lambda x: [word for word in x if word.isalnum()])
print(df['no_punct'].head())

# 7. Stop Words Removal
stop_words = set(stopwords.words('english'))
df['clean_tokens'] = df['no_punct'].apply(lambda x: [word for word in x if word not in stop_words])
print(df['clean_tokens'].head())

# Join tokens back to string
df['clean_text'] = df['clean_tokens'].apply(lambda x: " ".join(x))
print(df[['clean_text']].head())

# ── Task 2: TF-IDF + Naive Bayes ─────────────────────────────────────────────

# 8. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_text'])

# 9. Naive Bayes
y = df['label'].apply(lambda x: 1 if x == "Fake" else 0).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# 10. Model Evaluation
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Naive Bayes")
plt.show()

# Class Distribution Plot
counts = df['label'].value_counts().reindex(['Real', 'Fake'])
plt.figure()
counts.plot(kind='bar', color=['steelblue', 'orange'])
plt.xlabel('News Type')
plt.ylabel('Number of Articles')
plt.title('Real vs Fake News Counts')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
