import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from feature_selectors import select_k_best_feature_selection_v1, select_k_best_feature_selection_v2, cfs_feature_selection, rfe_feature_selector, rfecv_feature_selector, pca_feature_selector, permutation_importance_selector
import seaborn as sns
import matplotlib.pyplot as plt


def prepare_data(csv_file, audio_source, classify, algorithm=None, feature_selection=1, modalities=None, num_features=None, conditions_to_remove=None, column_name=None):
    """
    Prepare the data by loading, cleaning, and selecting features.

    Args:
        csv_file (str): The path to the csv file containing the data.
        audio_source (str): The name of the audio source.
        classify (str): The column name of the target variable to classify.
        algorithm (str): The algorithm to use for feature selection. Default is None.
                         'rf' for RandomForest, 'knn' for KNeighbors, 'mlp' for MLPClassifier
        feature_selection (int): The feature selection method to use. Default is 1.
        modalities (list): A list of modalities to consider. Default is None.
        num_features (int): The number of features to select. Default is None.
        conditions_to_remove (list): The conditions to remove from the dataset. Default is None.
        column_name (str): The name of the column to check for conditions to remove. Default is None.

    Returns:
        x (DataFrame): The prepared feature set.
        y (Series): The target variable.
        selected_features (list): The selected features (if feature selection is applied).
    """
    data = pd.read_csv(csv_file)

    metadata_columns = [
        "recording_condition",
        "phrase",
        "clip_number",
        "phonation",
        f"{audio_source}_note",
    ]

    x = data.drop(columns=metadata_columns)

    if modalities:
        x = extract_modality(modalities, x)

    x = handle_missing_data(x)
    y = data[classify]

    if conditions_to_remove:
        if column_name:
            x, y = remove_specified_conditions(x, y, conditions_to_remove, column_name)
            for condition in y:
                if condition in conditions_to_remove:
                    raise ValueError(f"Condition {condition} was not removed")
        else:
            raise ValueError("Column name must be provided when removing conditions")

    selected_features = None
    if num_features:
        if feature_selection == 1:
            x, selected_features = select_k_best_feature_selection_v1(num_features, x, y)
        elif feature_selection == 2:
            if algorithm:
                x, selected_features = select_k_best_feature_selection_v2(num_features, x, y, algorithm)
            else:
                raise ValueError("Algorithm type must be inputted with feature selector 2")
        elif feature_selection == 3:
            #cfs feature selection
            x, selected_features = cfs_feature_selection(x, y, num_features)
        elif feature_selection == 4:
            if algorithm:
                x, selected_features = rfecv_feature_selector(x, y, min_features_to_select=num_features, algorithm=algorithm)
            else:
                raise ValueError("Algorithm type must be inputted with feature selector 4")
        elif feature_selection == 5:
            x, selected_features = pca_feature_selector(x, y, num_features=num_features)
        elif feature_selection == 6:
            x, selected_features = permutation_importance_selector(x, y, num_features=num_features, algorithm=algorithm)
        elif feature_selection == 7:
          x, selected_features = rfe_feature_selector(x, y, num_features=num_features, algorithm=algorithm)
        else:
            raise ValueError("Inputted feature selection option not supported")
        print("Selected features: ", selected_features)

    return x, y, selected_features


def handle_missing_data(x):
    """
    Handle missing data by filling NaN values with zeros and ensuring consistent handling
    of missing pose landmarks across conditions. Pose landmakrs with missing data need to be
    handled differently to accommodate for possible different camera angles.

    Args:
        x (DataFrame): The input features.

    Returns:
        x (DataFrame): The input features with missing data handled.
    """
    # Identify columns with pose landmarks --
    pose_landmark_columns = [col for col in x.columns if "pose_landmark" in col]
    # Identify columns with NaNs before filling them
    nan_columns = x[pose_landmark_columns].isna().any(axis=0)

    # Replace all NaN values with zeros
    x = x.fillna(0)

    # Ensure consistent handling of missing pose landmarks across conditions
    for col in nan_columns.index:
        if nan_columns[col]:
            parts = col.split("_")
            landmark_number = parts[-2]  # Extract landmark number
            source = parts[0]  # Extract source (e.g., computer)
            columns_to_update = [
                f"{source}_pose_landmark_{landmark_number}_{axis}"
                for axis in ["x", "y", "z"]
            ]
            x[columns_to_update] = 0
    return x


def extract_modality(modalities, x):
    """
    Extract the specified modalities from the feature set.

    Args:
        modalities (list): The modalities to extract.
        x (DataFrame): The input features.

    Returns:
        x (DataFrame): The input features with the specified modalities extracted.
    """
    # Define columns to be dropped for each modality
    audio_columns_to_drop = [
        col for col in x.columns if any(substr in col for substr in ["spec", "mfcc", "tristimulus", "rms", "pitch"])
    ]
    video_columns_to_drop = [col for col in x.columns if "landmark" in col]
    biosignal_columns_to_drop = ["emg_1", "respiration_1"]

    # Check which modalities are not in the list and prepare columns to drop
    if "audio" not in modalities:
        x = x.drop(columns=audio_columns_to_drop)
    if "video" not in modalities:
        x = x.drop(columns=video_columns_to_drop)
    if "biosignals" not in modalities:
        x = x.drop(columns=biosignal_columns_to_drop)
    
    return x


def remove_specified_conditions(x, y, conditions_to_remove, column_name):
    """
    Remove specified conditions from the dataset based on a given column.

    Args:
        x (DataFrame): The input features.
        y (Series): The target variable.
        conditions_to_remove (list): The conditions to remove.
        column_name (str): The name of the column to check for conditions to remove.

    Returns:
        x (DataFrame): The input features with the specified conditions removed.
        y (Series): The target variable with the specified conditions removed.
    """
    # Concatenate x and y to ensure they stay aligned during filtering
    data = pd.concat([x, y], axis=1)

    for condition in conditions_to_remove:
        data = data[data[column_name] != condition]

    x_filtered = data.drop(columns=[column_name])
    y_filtered = data[y.name]

    return x_filtered, y_filtered


def standardize_x_data(X_train, X_test):
    """
    Standardizes the features in the train and test datasets separately.

    Args:
        X_train (DataFrame): Training feature set.
        X_test (DataFrame): Testing feature set.

    Returns:
        X_train_std (DataFrame): Standardized training feature set.
        X_test_std (DataFrame): Standardized testing feature set.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.
    """
    scaler = StandardScaler()

    X_train_std = scaler.fit_transform(X_train)
    X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)

    X_test_std = scaler.transform(X_test)
    X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)

    return X_train_std, X_test_std


def select_features(x, selected_features):
    '''
    Given a source, keep the columns that correspond to that source for the bio data.

    Args:
        x (DataFrame): The input features. 
        selected_features (list): List of selected features to use for model.

    Returns:
        x_new (DataFrame): The input features with the best selected features.
    '''
    # Convert all selected features to lowercase for case-insensitive matching
    selected_features_lower = [feature.lower() for feature in selected_features]
    
    # Filter columns that are in the selected_features list (case-insensitive)
    x_new = x[[col for col in x.columns if col.lower() in selected_features_lower]]
    
    return x_new


def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plot the confusion matrix as a heatmap.

    Args:
        conf_matrix (array): The confusion matrix to plot.
        class_names (array): The class names.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def calculate_metrics_2_class(conf_matrix):
    """
    Calculate precision, recall, specificity, and false positive rate from the confusion matrix.

    Args:
        conf_matrix (array): The confusion matrix.

    Returns:
        precision (float): The precision of the model.
        recall (float): The recall of the model.
        specificity (float): The specificity of the model.
        fpr (float): The false positive rate of the model.
    """
    TN, FP, FN, TP = conf_matrix.ravel()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    fpr = FP / (FP + TN)

    return precision, recall, specificity, fpr


def calculate_metrics_multi_class(y_true, y_pred):
    """
    Calculate precision and recall from the predictions and true labels.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        conf_matrix (array, optional): Confusion matrix. If provided, specificity and FPR will be calculated from it.

    Returns:
        precision (float): Precision score.
        recall (float): Recall score.
    """
    # Ensure y_true and y_pred are numpy arrays if they are pandas Series
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()

    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    return precision, recall
