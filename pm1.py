import json

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, max_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import joblib

from data_proccess import proccess_data


def get_data(file_name):
    df = pd.read_excel(file_name)
    return df


def prepare_data(data):
    """Transform categorical value 'I' into three boolean values"""
    data['I'] = pd.Categorical(data['I'])
    ohe = ColumnTransformer([("One hot", OneHotEncoder(sparse=False), [8])], remainder="passthrough")
    data = ohe.fit_transform(data)
    return data


def delete_model(dict):
    dict.pop('model')
    return dict


def model_process(model, X_train, X_test, y_train, y_test):
    """Train and estimate models, create dictionary to compare the quality of the model"""
    model.fit(X_train, y_train)
    y_model = model.predict(X_test)
    return {
        'MSE': mean_squared_error(y_test, y_model),
        'MAE': mean_absolute_error(y_test, y_model),
        'MAPE': mean_absolute_percentage_error(y_test, y_model),
        'max_err': max_error(y_test, y_model),
        'model': model,
        'model_name': str(model.estimator)[:-2]
    }


def prepare_models(X_train, y_train, X_test, y_test):
    """
    Iterating through the hyperparameters, find the best model, train it, predict target values for test data-set,
    compute the quality parameters of the model and append the parameters dictionary to the list.
    Here we use KNeighborsRegressor, Ridge, DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor fuctions for
    models training
    """
    models_info = []

    cv_knn_regr = GridSearchCV(KNeighborsRegressor(), param_grid={'n_neighbors': range(1, 15),
                                                                  'weights': ['uniform', 'distance'],
                                                                  'p': range(1, 4)})
    models_info.append(model_process(cv_knn_regr, X_train, y_train, X_test, y_test))
    cv_ridge = GridSearchCV(Ridge(), param_grid={'alpha': np.linspace(0.1, 3, 10)})
    models_info.append(model_process(cv_ridge, X_train, y_train, X_test, y_test))
    cv_decision_tree = GridSearchCV(DecisionTreeRegressor(random_state=10), param_grid={'max_depth': range(1, 10, 1)})
    models_info.append(model_process(cv_decision_tree, X_train, y_train, X_test, y_test))
    cv_random_tree = GridSearchCV(RandomForestRegressor(), param_grid={'n_estimators': range(60, 100, 10),
                                                                       'max_depth': range(50, 100, 10)})
    models_info.append(model_process(cv_random_tree, X_train, y_train, X_test, y_test))
    cv_extra_forest = GridSearchCV(ExtraTreesRegressor(), param_grid={'n_estimators': range(70, 90, 3),
                                                                      'max_depth': range(50, 70, 3)})
    models_info.append(model_process(cv_extra_forest, X_train, y_train, X_test, y_test))
    # Find the best models by sorting them by the value of Mean absolute percentage error (MAPE)
    models_info.sort(key=lambda x: x['MAPE'])
    return models_info


if __name__ == '__main__':
    file = 'data.xlsx'
    raw_data = get_data(file)
    raw_data = pd.concat([raw_data[180:280], raw_data[370:470], raw_data[1000:1100], raw_data[1540:1640]])
    data = prepare_data(raw_data)
    # Separating variables from the target values
    X = data[:, :-1]
    y = data[:, -1]
    # Divide variables and target values into a training and test sample
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    models_info = prepare_models(X_train, X_test, y_train, y_test)

    # Predict target value for the entire dataset, write new data to the DateFrame, save model object in .plk format
    for i in range(0, 3):
        raw_data[models_info[i]['model_name']] = models_info[i]['model'].predict(X)
        joblib.dump(models_info[i]['model'], f"{str(models_info[i]['model'].estimator)[:-2]}.plk")
    raw_data.to_excel('test_result.xlsx')
    models_info = list(map(delete_model, models_info))

    with open('models.json', 'w') as file:
        json.dump(models_info, file)
    # Write new data in database
    proccess_data(raw_data)
