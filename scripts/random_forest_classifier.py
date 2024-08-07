import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, classification_report

def train_random_forest(x_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier on the data.

    Args:
        x_train (DataFrame): The training feature set.
        y_train (Series): The training target variable.
        n_estimators (int): The number of trees in the forest. Default is 100.
        random_state (int): The random state for reproducibility. Default is 42.

    Returns:
        pipeline (Pipeline): The trained pipeline including scaler and Random Forest classifier.
    """
    # Random Forest classifier
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Pipeline with scaler and classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', rf)
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=10)

    # Perform cross-validation on the training set
    cv_scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring='f1_weighted')
    print(f"Mean cross-validated F1 score: {cv_scores.mean()}")

    pipeline.fit(x_train, y_train)

    return pipeline

def evaluate_random_forest(pipeline, x_test, y_test, class_names):
    '''
    Evaluate the performance of the Random Forest classifier on the test set.

    Args:
        pipeline (Pipeline): The trained pipeline including scaler and Random Forest classifier.
        x_test (DataFrame): The test feature set.
        y_test (Series): The true labels for the test set.
        class_names (list): The names of the classes.

    Returns:
        f1 (float): The F1 score of the model on the test set.
        conf_matrix (array): The confusion matrix of the model.
    '''
    y_pred = pipeline.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=class_names)

    return f1, class_report, conf_matrix
