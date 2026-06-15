import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys


def find_text_label_columns(df):
    cols = list(df.columns)
    lowered = [c.lower() for c in cols]

    # prefer columns that contain 'resume'
    text_candidates = [c for c in cols if 'resume' in c.lower() or 'cv' in c.lower()]
    # fallback to common names
    if not text_candidates:
        text_candidates = [c for c in cols if any(k in c.lower() for k in ("text","description","content"))]

    label_candidates = [c for c in cols if any(k in c.lower() for k in ("category","label","target","job","role","class"))]

    text_col = text_candidates[0] if text_candidates else None
    label_col = label_candidates[0] if label_candidates else None

    if text_col is None or label_col is None:
        # fallback: if exactly 2 columns, assume first=text second=label
        if len(cols) == 2:
            text_col, label_col = cols[0], cols[1]

    return text_col, label_col


def main():
    csv_path = "Resume.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        sys.exit(1)

    text_col, label_col = find_text_label_columns(df)
    if text_col is None or label_col is None:
        print("Could not automatically detect text and label columns. Columns found:")
        print(df.columns.tolist())
        sys.exit(1)

    df = df[[text_col, label_col]].dropna()
    X = df[text_col].astype(str).values
    y = df[label_col].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y if len(set(y))>1 else None)

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    clf = LogisticRegression(max_iter=1000)

    pipeline = Pipeline([('vect', vectorizer), ('clf', clf)])
    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test) if len(X_test)>0 else None
    print(f"Validation accuracy: {score}")

    # Save vectorizer and model separately for compatibility with the Streamlit app
    # The app expects `vectorizer.pkl` and `resume_model.pkl` where model has a predict method taking transformed vectors.
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save the classifier alone (not the pipeline) so app can call vectorizer.transform then model.predict
    with open('resume_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print('Saved vectorizer.pkl and resume_model.pkl')


if __name__ == '__main__':
    main()
