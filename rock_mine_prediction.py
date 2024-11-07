import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_dataset(csv_file_path:str) -> pd.DataFrame:
    """
    This function loads a csv file.

    Args: 
        csv_file_path: this is the path to the 
        csv file to be loaded
    Returns: Pandas DataFrame of the loaded csv file.
    """
    pandas_data = pd.read_csv(csv_file_path, header=None)
    return pandas_data

def process_data(dataframe: pd.DataFrame):
    """
    This function processes the data by splitting it into features and target.

    Args:
    dataframe: the data to be processed

    Returns:
    features and target
    """
    features = dataframe.iloc[:, :-1]
    target = dataframe.iloc[:, -1]
    return features, target

def load_and_train_model(features, labels):
    """
    This function loads the data, trains a model and makes predictions.
    Args:
    features: the features of the data
    labels: the labels of the data
    Returns:
    accuracy of the model
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, stratify=labels, random_state=1)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    training_prediction = model.predict(X_train)
    training_accuracy = accuracy_score(training_prediction, y_train)

    test_prediction = model.predict(X_test)
    test_accuracy = accuracy_score(test_prediction, y_test)

    print(f"Training accuracy: {training_accuracy}")
    print(f"Test accuracy: {test_accuracy}")
    return model

def make_predictions(model, input_data:tuple):
    """
    This function makes predictions using the trained model.
    Args:
    model: the trained model
    input_data: the data to be predicted
    Returns:
    prediction
    """
    input_array = np.asarray(input_data)
    reshapped_array = input_array.reshape(1, -1)
    prediction = model.predict(reshapped_array)
    if prediction[0] == 'R':
        print("Object is a rock.")
    else:
        print("Object is a mine.")


if __name__ == "__main__":
    dataframe = load_dataset('Copy of sonar data.csv')
    features, labels = process_data(dataframe)
    model = load_and_train_model(features, labels)
    make_predictions(model, (0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032))
