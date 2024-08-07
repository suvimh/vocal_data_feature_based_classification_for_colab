import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance


def select_k_best_feature_selection_v1(num_features, x, y):
    """
    Select the top num_features using the SelectKBest method.

    Args:
        num_features (int): The number of features to select.
        X (DataFrame): The input features.
        y (Series): The target variable.

    Returns:
        X_new (DataFrame): The selected features.
        selected_features (list): The names of the selected features.
    """
    # Drop constant columns before feature selection
    constant_columns = [col for col in x.columns if x[col].nunique() == 1]
    x = x.drop(columns=constant_columns)

    selector = SelectKBest(f_classif, k=num_features)
    x_new = selector.fit_transform(x, y)
    selected_mask = selector.get_support()
    selected_features = x.columns[selected_mask]

    return x_new, selected_features


def select_k_best_feature_selection_v2(max_features, x, y, algorithm, hidden_layers=(12,), random_state=42):
    """
    Select the best performing feature combination up to max_features using SelectKBest and cross-validation.

    Args:
        max_features (int): The maximum number of features to select.
        x (DataFrame): The input features.
        y (Series): The target variable.
        algorithm (str): The algorithm to evaluate ('knn', 'mlp', or 'rf').
        hidden_layers (tuple): The number of hidden layers and units in each layer for the MLP classifier.
        random_state (int): The random state for reproducibility.

    Returns:
        x_new (DataFrame): The input features with the best selected features.
        selected_features (list): The names of the selected features.
    """
    # Drop constant columns before feature selection
    constant_columns = [col for col in x.columns if x[col].nunique() == 1]
    x = x.drop(columns=constant_columns)
    
    best_score = -float('inf')
    selected_features = None
    
    # Iterate over number of features to select
    for k in range(1, max_features + 1):
        selector = SelectKBest(f_classif, k=k)
        x_new = selector.fit_transform(x, y)
        selected_mask = selector.get_support()
        current_selected_features = x.columns[selected_mask]

        # Evaluate the performance using cross-validation for the specified algorithm
        if algorithm == 'knn':
            model = KNeighborsClassifier()
        elif algorithm == 'mlp':
            model = MLPClassifier(hidden_layer_sizes=hidden_layers,
                                  learning_rate_init=0.01,
                                  momentum=0.5,
                                  max_iter=2000,
                                  random_state=random_state)
        elif algorithm == 'rf':
            model = RandomForestClassifier(random_state=random_state)
        else:
            raise ValueError("Unsupported algorithm. Choose 'knn', 'mlp', or 'rf'.")

        scores = cross_val_score(model, x_new, y, cv=5)
        mean_score = scores.mean()
        
        # Update best score and features if current combination is better
        if mean_score > best_score:
            best_score = mean_score
            selected_features = current_selected_features
    
    # Select the best features
    x_new = x[selected_features]
    
    return x_new, list(selected_features)

def get_model_from_string(algorithm):
    """
    Returns a model based on the input string.

    Parameters:
    - algorithm: str
        The algorithm identifier ('rf', 'knn', 'mlp').

    Returns:
    - model: object
        The instantiated model corresponding to the algorithm identifier.
    """
    if algorithm == 'rf':
        return RandomForestClassifier()
    elif algorithm == 'knn':
        return KNeighborsClassifier()
    elif algorithm == 'mlp':
        return MLPClassifier()
    else:
        raise ValueError("Unsupported algorithm. Choose 'rf', 'knn', or 'mlp'.")

def cfs_feature_selection(x, y, num_features):
    """
    Perform Correlation-based Feature Selection (CFS).

    Args:
        x (DataFrame): The feature matrix.
        y (Series): The target variable.
        num_features (int): The number of features to select.

    Returns:
        x_selected (DataFrame): Selected feature matrix.
        selected_features (list): List of selected feature names.
    """
    # Convert y to numeric labels if necessary
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    # Calculate feature-class correlations
    feature_class_corr = []
    for feature in x.columns:
        corr = np.abs(np.corrcoef(x[feature], y)[0, 1])  # Absolute correlation with class
        feature_class_corr.append((feature, corr))

    # Sort features by correlation with class label (descending)
    feature_class_corr.sort(key=lambda x: x[1], reverse=True)

    # Initialize selected features list and feature set
    selected_features = []
    feature_set = set()

    # Add the first feature (highest correlation with class)
    selected_features.append(feature_class_corr[0][0])
    feature_set.add(feature_class_corr[0][0])

    # Iterate until desired number of features is selected
    while len(selected_features) < num_features:
        max_cfs = -float('inf')
        best_feature = None

        # Evaluate CFS criterion for each remaining feature
        for feature, _ in feature_class_corr:
            if feature not in feature_set:
                new_feature_set = feature_set.union([feature])
                cfs_value = calculate_cfs(x[list(new_feature_set)], y)

                # Update best feature based on CFS value
                if cfs_value > max_cfs:
                    max_cfs = cfs_value
                    best_feature = feature

        # Add the best feature to the selected features list and set
        selected_features.append(best_feature)
        feature_set.add(best_feature)

    # Return selected features and corresponding feature matrix
    x_selected = x[selected_features]
    return x_selected, selected_features

def calculate_cfs(x_subset, y):
    """
    Calculate the CFS value for a subset of features.

    Args:
        x_subset (DataFrame): Subset of feature matrix.
        y (Series): The target variable.

    Returns:
        cfs_value (float): CFS criterion value.
    """
    # Calculate feature-feature correlations
    correlations = x_subset.corr().abs()

    # Calculate feature-class correlations
    f_class = np.abs(np.corrcoef(x_subset.T, y)[0, 1:])

    # Calculate CFS criterion
    num_features = len(x_subset.columns)
    cfs_value = (np.mean(f_class) / ((1 / num_features) * np.sum(np.sum(correlations)))) * num_features

    return cfs_value


def rfecv_feature_selector(x, y, min_features_to_select, algorithm):
    """
    Perform feature selection using RFECV.

    Parameters:
    - x: DataFrame or array-like, shape (n_samples, n_features)
         The input samples.
    - y: array-like, shape (n_samples,)
         The target values.
    - min_features_to_select: int
         The number of top features to select.
    - algorithm: str
         The algorithm to use ('rf' for RandomForest, 'knn' for KNeighbors, 'mlp' for MLPClassifier).

    Returns:
    - x_transformed: DataFrame or array-like, shape (n_samples, min_features_to_select)
         The input samples with selected features.
    - selected_features: list
         List of selected feature names.
    """
    
    model = get_model_from_string(algorithm)

    # Perform RFECV
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='accuracy', min_features_to_select=min_features_to_select)
    rfecv.fit(x, y)
    
    # Get selected features
    selected_features = x.columns[rfecv.support_]
    
    # Transform the dataset
    x_transformed = rfecv.transform(x)
    
    return x_transformed, selected_features


def pca_feature_selector(x, y, num_features):
    """
    Perform feature selection using PCA.

    Parameters:
    - x: DataFrame or array-like, shape (n_samples, n_features)
         The input samples.
    - y: array-like, shape (n_samples,)
         The target values.
    - num_features: int
         The number of top features to select.

    Returns:
    - x_transformed: DataFrame or array-like, shape (n_samples, num_features)
         The input samples with selected features.
    - selected_features: list
         List of selected feature names (in this case, the principal components).
    """
    
    pca = PCA(n_components=num_features)
    x_transformed = pca.fit_transform(x)
    
    selected_features = [f"PC{i+1}" for i in range(num_features)]
    
    return x_transformed, selected_features


def permutation_importance_selector(fitted_model, x, y, num_features):
    """
    Perform feature selection using Permutation Importance.

    Parameters:
    - fitted_model: The pre-fitted model to be used for permutation importance.
    - x: DataFrame or array-like, shape (n_samples, n_features)
         The input samples.
    - y: array-like, shape (n_samples,)
         The target values.
    - num_features: int
         The number of top features to select.

    Returns:
    - x_transformed: DataFrame or array-like, shape (n_samples, num_features)
         The input samples with selected features.
    - selected_features: list
         List of selected feature names.
    """
    
    # Compute permutation importance
    result = permutation_importance(fitted_model, x, y, n_repeats=10, random_state=42, scoring='accuracy')
    importance = result.importances_mean
    
    # Get indices of top features
    indices = np.argsort(importance)[-num_features:]
    
    # Get the names of the top features
    selected_features = x.columns[indices]
    
    # Transform the dataset
    x_transformed = x[selected_features]
    
    return x_transformed, selected_features