import pickle
import os

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import ComplementNB

DATA_PATH = 'data/messages_clean.csv'
MODEL_PATH = 'models/nb_pipeline.pkl'


def main():
    print('Loading data...')
    df = pd.read_csv(DATA_PATH)
    X, y = df['clean_content'], df['author']
    print(f'  {len(df)} messages, {y.nunique()} authors')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print('Vectorising...')
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print('Running GridSearchCV (ComplementNB, alpha in [0.01, 0.1, 0.5, 1.0], 5-fold CV)...')
    gs = GridSearchCV(
        ComplementNB(),
        {'alpha': [0.01, 0.1, 0.5, 1.0]},
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train_tfidf, y_train)
    print(f'  Best alpha: {gs.best_params_["alpha"]}')
    print(f'  Best CV accuracy: {gs.best_score_:.4f}')

    print('Calibrating model (isotonic, cv=5)...')
    model = CalibratedClassifierCV(gs.best_estimator_, method='isotonic', cv=5)
    model.fit(X_train_tfidf, y_train)

    test_acc = accuracy_score(y_test, model.predict(X_test_tfidf))
    print(f'  Test accuracy: {test_acc:.4f}')

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'model': model}, f)
    print(f'Saved {MODEL_PATH}')


if __name__ == '__main__':
    main()
