"""
Final Project – Bug Cases
AI 100 | Sentiment Classification (MLP + TF-IDF)
Each function demonstrates one intentional bug introduced into the midterm code.
"""

import random, re, numpy as np
from collections import Counter

random.seed(42)
np.random.seed(42)

TEMPLATES = {
    0: ["This product is terrible and completely broken.",
        "I hate this item, it stopped working after one day.",
        "Worst purchase I have ever made, total waste of money.",
        "The quality is awful and the customer service is horrible.",
        "Extremely disappointed, do not buy this garbage.",
        "Broke immediately, feel totally ripped off.",
        "Nothing works as advertised, completely useless junk.",
        "Really bad experience, poor build quality and cheap materials.",
        "The item arrived damaged and smelled awful.",
        "Terrible product, would never recommend to anyone.",
        "Does not function at all, major disappointment.",
        "Cheap, flimsy, and falls apart instantly."],
    1: ["The product works as expected, nothing special.",
        "It is okay, does what it is supposed to do.",
        "Average quality, meets basic requirements.",
        "Decent item for the price, nothing extraordinary.",
        "It arrived on time and functions adequately.",
        "Not bad but not great, fairly standard product.",
        "Works fine, no complaints but no praise either.",
        "Acceptable quality, does the job without issues.",
        "Mediocre performance, meets minimum expectations.",
        "An ordinary product, neither impressive nor disappointing.",
        "It is functional but unremarkable in every way.",
        "Pretty standard item, nothing to write home about."],
    2: ["This product is absolutely amazing and exceeds expectations!",
        "I love this item, it works perfectly every single time.",
        "Best purchase I have ever made, highly recommend!",
        "Fantastic quality and incredible customer service.",
        "Extremely satisfied, this product changed my life.",
        "Works flawlessly, so happy with this great purchase.",
        "Everything works perfectly, outstanding quality and value.",
        "Really excellent experience, superb build and great materials.",
        "The item arrived quickly and works brilliantly.",
        "Wonderful product, would recommend to absolutely everyone.",
        "Performs beyond expectations, truly impressive item.",
        "High quality, durable, and absolutely worth every penny."],
}

def generate_dataset(n=600):
    texts, labels = [], []
    per_class = n // 3
    for label, sentences in TEMPLATES.items():
        for _ in range(per_class):
            base = random.choice(sentences)
            words = base.split()
            if random.random() < 0.3 and len(words) > 5:
                idx = random.randint(1, len(words) - 2)
                words.insert(idx, random.choice(["very", "quite", "really", "honestly"]))
            texts.append(" ".join(words))
            labels.append(label)
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)

STOPWORDS = {
    "i","me","my","we","our","you","your","he","she","it","its","they","their",
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","this","that","these",
    "those","am","not","no","so","as","by","from","about","up","out","if","than",
}

def preprocess_fallback(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


# ─────────────────────────────────────────────
# CASE 1 – Wrong random seed (student: Alex)
# Original: random.seed(42) / np.random.seed(42)
# Bug: random.seed(0) / np.random.seed(0)
# Effect: Dataset shuffling changes → different train/test split,
#         results are no longer reproducible with seed 42.
# ─────────────────────────────────────────────
def case1_wrong_seed():
    random.seed(0)          # BUG: was 42
    np.random.seed(0)       # BUG: was 42
    texts, labels = generate_dataset(600)
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier

    processed = [" ".join(preprocess_fallback(t)) for t in texts]
    X_traindev, X_test, y_traindev, y_test = train_test_split(
        processed, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_traindev, y_traindev, test_size=0.1765, random_state=42, stratify=y_traindev)
    vec = TfidfVectorizer(max_features=500)
    Xtr = vec.fit_transform(X_train); Xdev = vec.transform(X_dev); Xte = vec.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=300, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    clf.fit(Xtr, y_train)
    print(f"[Case 1] Dev acc: {clf.score(Xdev, y_dev):.4f}  Test acc: {clf.score(Xte, y_test):.4f}")


# ─────────────────────────────────────────────
# CASE 2 – generate_dataset called with n=0 (student: Alex)
# Original: generate_dataset(600)
# Bug: generate_dataset(0)
# Effect: per_class = 0 → no samples generated → empty lists → crash at zip(*combined)
# ─────────────────────────────────────────────
def case2_empty_dataset():
    texts, labels = generate_dataset(0)   # BUG: was 600


# ─────────────────────────────────────────────
# CASE 3 – Stopword set replaced with empty set (student: Alex)
# Original: STOPWORDS = { "i","me","my", ... }
# Bug: STOPWORDS = set()
# Effect: No words removed → vocabulary bloated → noisier features,
#         the model may still run but generalization suffers.
# ─────────────────────────────────────────────
def case3_empty_stopwords():
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier

    texts, labels = generate_dataset(600)

    def preprocess_no_stop(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        return text.split()   # BUG: STOPWORDS check removed (empty set effectively)

    processed = [" ".join(preprocess_no_stop(t)) for t in texts]
    X_traindev, X_test, y_traindev, y_test = train_test_split(
        processed, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_traindev, y_traindev, test_size=0.1765, random_state=42, stratify=y_traindev)
    vec = TfidfVectorizer(max_features=500)
    Xtr = vec.fit_transform(X_train); Xdev = vec.transform(X_dev); Xte = vec.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=300, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    clf.fit(Xtr, y_train)
    print(f"[Case 3] Dev acc: {clf.score(Xdev, y_dev):.4f}  Test acc: {clf.score(Xte, y_test):.4f}")


# ─────────────────────────────────────────────
# CASE 4 – TF-IDF fitted on TEST data (student: Alex)
# Original: vectorizer.fit_transform(X_train)
# Bug: vectorizer.fit_transform(X_test)  then transform(X_train)
# Effect: Data leakage — vocabulary built from test set.
#         X_train transformed using test-set vocabulary → feature mismatch / inflated metrics.
# ─────────────────────────────────────────────
def case4_tfidf_fit_on_test():
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier

    texts, labels = generate_dataset(600)
    processed = [" ".join(preprocess_fallback(t)) for t in texts]
    X_traindev, X_test, y_traindev, y_test = train_test_split(
        processed, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_traindev, y_traindev, test_size=0.1765, random_state=42, stratify=y_traindev)

    vec = TfidfVectorizer(max_features=500)
    Xte  = vec.fit_transform(X_test)    # BUG: fit on test, was X_train
    Xtr  = vec.transform(X_train)
    Xdev = vec.transform(X_dev)

    clf = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=300, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    clf.fit(Xtr, y_train)
    print(f"[Case 4] Dev acc: {clf.score(Xdev, y_dev):.4f}  Test acc: {clf.score(Xte, y_test):.4f}")


# ─────────────────────────────────────────────
# CASE 5 – max_features=1 in TF-IDF (student: Alex)
# Original: TfidfVectorizer(max_features=500)
# Bug: TfidfVectorizer(max_features=1)
# Effect: Only ONE feature kept → model has almost no signal → near-random accuracy.
# ─────────────────────────────────────────────
def case5_tfidf_one_feature():
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier

    texts, labels = generate_dataset(600)
    processed = [" ".join(preprocess_fallback(t)) for t in texts]
    X_traindev, X_test, y_traindev, y_test = train_test_split(
        processed, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_traindev, y_traindev, test_size=0.1765, random_state=42, stratify=y_traindev)

    vec = TfidfVectorizer(max_features=1)   # BUG: was 500
    Xtr = vec.fit_transform(X_train); Xdev = vec.transform(X_dev); Xte = vec.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=300, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    clf.fit(Xtr, y_train)
    print(f"[Case 5] Dev acc: {clf.score(Xdev, y_dev):.4f}  Test acc: {clf.score(Xte, y_test):.4f}")


# ─────────────────────────────────────────────
# CASE 6 – activation changed to 'identity' (student: Jamie)
# Original: activation="relu"
# Bug: activation="identity"
# Effect: Every hidden layer becomes a pure linear transform → network collapses
#         to a linear model regardless of depth; accuracy drops noticeably.
# ─────────────────────────────────────────────
def case6_linear_activation():
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier

    texts, labels = generate_dataset(600)
    processed = [" ".join(preprocess_fallback(t)) for t in texts]
    X_traindev, X_test, y_traindev, y_test = train_test_split(
        processed, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_traindev, y_traindev, test_size=0.1765, random_state=42, stratify=y_traindev)
    vec = TfidfVectorizer(max_features=500)
    Xtr = vec.fit_transform(X_train); Xdev = vec.transform(X_dev); Xte = vec.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(256,128,64),
        activation="identity",   # BUG: was "relu"
        max_iter=300, random_state=42, early_stopping=True, validation_fraction=0.1)
    clf.fit(Xtr, y_train)
    print(f"[Case 6] Dev acc: {clf.score(Xdev, y_dev):.4f}  Test acc: {clf.score(Xte, y_test):.4f}")


# ─────────────────────────────────────────────
# CASE 7 – max_iter=1 (student: Jamie)
# Original: max_iter=300
# Bug: max_iter=1
# Effect: Training stops after one iteration; ConvergenceWarning raised;
#         model severely undertrained → low accuracy.
# ─────────────────────────────────────────────
def case7_one_iteration():
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier

    texts, labels = generate_dataset(600)
    processed = [" ".join(preprocess_fallback(t)) for t in texts]
    X_traindev, X_test, y_traindev, y_test = train_test_split(
        processed, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_traindev, y_traindev, test_size=0.1765, random_state=42, stratify=y_traindev)
    vec = TfidfVectorizer(max_features=500)
    Xtr = vec.fit_transform(X_train); Xdev = vec.transform(X_dev); Xte = vec.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu",
                        max_iter=1,            # BUG: was 300
                        random_state=42, early_stopping=True, validation_fraction=0.1)
    clf.fit(Xtr, y_train)
    print(f"[Case 7] Dev acc: {clf.score(Xdev, y_dev):.4f}  Test acc: {clf.score(Xte, y_test):.4f}")


# ─────────────────────────────────────────────
# CASE 8 – Labels shuffled independently from texts (student: Jamie)
# Original: combined = list(zip(texts, labels)); random.shuffle(combined)
# Bug: random.shuffle(texts); random.shuffle(labels)  (shuffled separately)
# Effect: texts and labels are mismatched → model trains on wrong signal → ~33% accuracy.
# ─────────────────────────────────────────────
def case8_label_mismatch():
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier

    texts, labels = generate_dataset(600)
    # BUG: shuffle separately instead of together
    random.shuffle(texts)    # BUG
    random.shuffle(labels)   # BUG (was: combined = list(zip(texts,labels)); random.shuffle(combined))

    processed = [" ".join(preprocess_fallback(t)) for t in texts]
    X_traindev, X_test, y_traindev, y_test = train_test_split(
        processed, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_traindev, y_traindev, test_size=0.1765, random_state=42, stratify=y_traindev)
    vec = TfidfVectorizer(max_features=500)
    Xtr = vec.fit_transform(X_train); Xdev = vec.transform(X_dev); Xte = vec.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=300, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    clf.fit(Xtr, y_train)
    print(f"[Case 8] Dev acc: {clf.score(Xdev, y_dev):.4f}  Test acc: {clf.score(Xte, y_test):.4f}")


# ─────────────────────────────────────────────
# CASE 9 – f1_score average changed to None → crash (student: Jamie)
# Original: f1_score(y_test, y_pred, average="macro")
# Bug: f1_score(y_test, y_pred, average=None)  → returns array, not scalar
#      then code tries to print it as a formatted float → TypeError
# ─────────────────────────────────────────────
def case9_f1_average_none():
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, f1_score

    texts, labels = generate_dataset(600)
    processed = [" ".join(preprocess_fallback(t)) for t in texts]
    X_traindev, X_test, y_traindev, y_test = train_test_split(
        processed, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_traindev, y_traindev, test_size=0.1765, random_state=42, stratify=y_traindev)
    vec = TfidfVectorizer(max_features=500)
    Xtr = vec.fit_transform(X_train); Xdev = vec.transform(X_dev); Xte = vec.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=300, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    macro_f1 = f1_score(y_test, y_pred, average=None)   # BUG: was "macro" → returns array
    print(f"Macro-F1: {macro_f1:.4f}")                  # TypeError: unsupported format character


# ─────────────────────────────────────────────
# CASE 10 – test_size=1.5 in train_test_split → ValueError (student: Jamie)
# Original: test_size=0.15
# Bug: test_size=1.5
# Effect: sklearn raises ValueError because test_size must be in (0,1) for float.
# ─────────────────────────────────────────────
def case10_invalid_test_size():
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts, labels = generate_dataset(600)
    processed = [" ".join(preprocess_fallback(t)) for t in texts]
    X_traindev, X_test, y_traindev, y_test = train_test_split(
        processed, labels, test_size=1.5,    # BUG: was 0.15
        random_state=42, stratify=labels)


if __name__ == "__main__":
    print("Running Case 1 (wrong seed)...")
    case1_wrong_seed()

    print("\nRunning Case 2 (empty dataset)...")
    try:
        case2_empty_dataset()
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\nRunning Case 3 (no stopwords)...")
    case3_empty_stopwords()

    print("\nRunning Case 4 (TF-IDF fit on test)...")
    case4_tfidf_fit_on_test()

    print("\nRunning Case 5 (TF-IDF max_features=1)...")
    case5_tfidf_one_feature()

    print("\nRunning Case 6 (identity activation)...")
    case6_linear_activation()

    print("\nRunning Case 7 (max_iter=1)...")
    case7_one_iteration()

    print("\nRunning Case 8 (label mismatch)...")
    case8_label_mismatch()

    print("\nRunning Case 9 (f1 average=None)...")
    try:
        case9_f1_average_none()
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\nRunning Case 10 (invalid test_size)...")
    try:
        case10_invalid_test_size()
    except Exception as e:
        print(f"  ERROR: {e}")
