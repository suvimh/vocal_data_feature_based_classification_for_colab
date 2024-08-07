import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,  f1_score
from utils import prepare_data


def train_and_test_knn(x_train, y_train, x_test, n_neighbors=5):
    '''
    Train and test a KNN classifier on the data in the csv file.    

    Args:
        x_train (DataFrame): The training feature set.
        y_train (Series): The training target variable.
        x_test (DataFrame): The testing feature set.
        n_neighbors (int): The number of neighbors to use for the KNN classifier. Default is 5.

    Returns:
        y_test (array): The true labels for the test set.
        y_pred (array): The predicted labels for the test set.
    '''
        
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    
    y_pred = knn.predict(x_test)
    
    return y_pred


def evaluate_model(y_test, y_pred):
    '''
        Evaluate the performance of the model using accuracy, F1-score, and confusion matrix.

        Args:
            y_test (array): The true labels for the test set.
            y_pred (array): The predicted labels for the test set.
        
        Returns:
            accuracy (float): The accuracy of the model.
            f1 (float): The F1-score of the model.
            conf_matrix (array): The confusion matrix of the model.
    '''
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Calculate F1-score
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, f1, conf_matrix