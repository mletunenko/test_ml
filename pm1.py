import json
import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, max_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from data_proccess import proccess_data


def get_data(file_name):
    """Read original data set from .xlsx file"""
    df = pd.read_excel(file_name)
    return df


def scale_dataset(data):
    """Scale to unit variance"""
    scaler = StandardScaler()
    scaler.fit(data.iloc[:, :-2])
    return scaler.transform(data.iloc[:, :-2])


def prepare_data(data):
    """Transform categorical value 'I' into three boolean values"""
    data['I'] = pd.Categorical(data['I'])
    ohe = ColumnTransformer([("One hot", OneHotEncoder(sparse=False), [8])], remainder="passthrough")
    data = ohe.fit_transform(data)
    return data[:, :3]


def delete_model(dict):
    """Delete model object from dictionary"""
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
        'model_name': str(model)[:-2],
        'score': model.score(X_test, y_test)
    }


def prepare_models(X_train, y_train, X_test, y_test):
    """
    Train models, predict target values for test data-set, compute the quality parameters of the model
    and append the parameters dictionary to the list.
    Here we use LinearRegression, Ridge, SGDRegressor, BayesianRidge fuctions for models training
    """
    # Create empty list to store dicts with information about models
    models_info = []
    # Create, train and estimate linear regression models
    reg = linear_model.LinearRegression()
    models_info.append(model_process(reg, X_train, y_train, X_test, y_test))
    ridge = linear_model.Ridge()
    models_info.append(model_process(ridge, X_train, y_train, X_test, y_test))
    sgd = linear_model.SGDRegressor()
    models_info.append(model_process(sgd, X_train, y_train, X_test, y_test))
    bayes = linear_model.BayesianRidge()
    models_info.append(model_process(bayes, X_train, y_train, X_test, y_test))
    # Find the best models by sorting them by the value of score
    models_info.sort(key=lambda x: - x['score'])
    return models_info


if __name__ == '__main__':
    file = 'data.xlsx'
    # Try to open dataset file or stop execution of program
    try:
        raw_data = get_data(file)
    except FileNotFoundError:
        print(f'Файл {file} не найден в директории проекта')
        exit(1)
    plt.figure(figsize=(20, 7))
    # Create a heatmap of correlation dependent variable and target value
    sns.heatmap(raw_data.corr(), annot=True, fmt='.2g', vmin=-1, vmax=1, center=0, cmap='coolwarm',
                cbar_kws={'orientation': 'horizontal',
                          'shrink': 0.75})
    # Save heatmap plot in .png format
    plt.savefig('Корреляция зависимых параметров с целевой функцией.png')
    # Use only 180-279, 370-469, 1000-1099, 1540-1639 raws from dataset for input data
    raw_data = pd.concat([raw_data[180:280], raw_data[370:470], raw_data[1000:1100], raw_data[1540:1640]])
    scaled_data = scale_dataset(raw_data)
    # Prepare data for correct processing of categorical value
    categorical_data = prepare_data(raw_data)
    # Get numpy array from the DataFrame
    grade = raw_data['grade'].to_numpy()
    # Reshape ndarray for concatenation
    grade.shape = (400, 1)
    # Concatenate sets with categorical data, scaled data and target value
    data = np.concatenate([categorical_data, scaled_data, grade], axis=1)
    # Separating variables from the target values
    X = data[:, :-1]
    y = data[:, -1]
    # Divide variables and target values into a training and test sample
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    # Train models and create sorted list of dictionaries with models parameters
    models_info = prepare_models(X_train, X_test, y_train, y_test)
    # Predict target value for the entire dataset, write new data to the DateFrame, save model object in .plk format
    for i in range(0, 3):
        raw_data[models_info[i]['model_name']] = models_info[i]['model'].predict(X)
        joblib.dump(models_info[i]['model'], f"{str(models_info[i]['model'])[:-2]}.plk")
    # Save new dataset with predicted target values
    raw_data.to_excel('test_result.xlsx')
    # Delete model object from dictionaries
    models_info = list(map(delete_model, models_info))
    # Write file with parameters of the models
    with open('models.json', 'w') as file:
        json.dump(models_info, file)
    # Write new data in database
    proccess_data(raw_data)
