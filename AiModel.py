import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# Load data into a pandas DataFrame
data =pd.read_csv('C:\Users\Joker\Downloads\containers.csv')


# Prepare the data by normalizing the input features
scaler = StandardScaler()
inputs = scaler.fit_transform(data.iloc[:, 2:10])



# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(inputs, data, test_size=0.2, random_state=42)



# Build a Random Forest regressor
rf_regressor = RandomForestRegressor()
rf_scores = cross_val_score(rf_regressor, x_train, y_train, cv=10,scoring='neg_mean_squared_error')
rf_mse_train = np.mean(rf_scores)
rf_mse_test = mean_squared_error(y_test, rf_regressor.fit(x_train, y_train).predict(x_test))


]
# Build a Neural Network regressor
nn_regressor = MLPRegressor()
nn_scores = cross_val_score(nn_regressor, x_train, y_train, cv=10)
nn_mse_train = np.mean(nn_scores)
nn_mse_test = mean_squared_error(y_test, nn_regressor.fit(x_train, y_train).predict(x_test))


# Build a Support Vector Machine regressor
svm_regressor = SVR()
svm_scores = cross_val_score(svm_regressor, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
svm_mse_train = np.mean(-svm_scores)
svm_mse_test = mean_squared_error(y_test, svm_regressor.fit(x_train, y_train).predict(x_test))


# Print the mean squared error rates
print("Random Forest MSE (Training):", rf_mse_train)
print("Random Forest MSE (Testing):", rf_mse_test)
print("Neural Network MSE (Training):", nn_mse_train)
print("Neural Network MSE (Testing):", nn_mse_test)
print("Support Vector Machine MSE (Training):", svm_mse_train)
print("Support Vector Machine MSE (Testing):", svm_mse_test)


# Plot the MSE results using a boxplot
plt.boxplot([nn_scores, svm_scores])
plt.xticks([1, 2, 3], ['Random Forest', 'Neural Network', 'Support Vector Machine'])
plt.ylabel('Negative Mean Squared Error')
plt.show()


# Predict the top 10 container ships by capacity for each regressor
rf_predictions = pd.DataFrame({'Name': data['ANNA MAERSK'], 'Capacity': rf_regressor.predict(inputs)})
rf_predictions = rf_predictions.sort_values(by=['Capacity'], ascending=False).head(10)
print("Random Forest Top 10 Predictions:\n", rf_predictions)
nn_predictions = pd.DataFrame({'Name': data['ANNA MAERSK'], 'Capacity': nn_regressor.predict(inputs)})
nn_predictions = nn_predictions.sort_values(by=['Capacity'], ascending=False).head(10)

#Task 1.2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


df = pd.read_csv('C:\Users\Joker\Downloads\containers.csv')

X = df.iloc[:, 3:11]
y = df.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
rf_predictions = rf_regressor.predict(X_test)
nn_regressor = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
nn_regressor.fit(X_train, y_train)
nn_predictions = nn_regressor.predict(X_test)
svm_regressor = SVR(kernel='rbf', gamma=0.1, C=10.0)
svm_regressor.fit(X_train, y_train)
svm_predictions = svm_regressor.predict(X_test)


f_df = pd.DataFrame({'Ship Name': df['ANNA MAERSK'], 'Predicted Capacity': rf_regressor.predict(X)})

nn_df = pd.DataFrame({'Ship Name': df['ANNA MAERSK'], 'Predicted Capacity': nn_regressor.predict(X)})
nn_df.sort_values('Predicted Capacity', ascending=False, inplace=True)
print(nn_df.head(10))

svm_df = pd.DataFrame({'Ship Name': df['ANNA MAERSK'], 'Predicted Capacity': svm_regressor.predict(X)})
svm_df.sort_values('Predicted Capacity', ascending=False, inplace=True)
print(svm_df.head(10))

#/#Task 1.3 – Assessment of regression
#To assess the regression models, we will use cross validation to report training and testing mean square error rates
#and use a boxplot to visualize the results:

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rf_scores = -cross_val_score(rf_regressor, X, y, cv=10, scoring='neg_mean_squared_error')
nn_scores = -cross_val_score(nn_regressor, X, y, cv=10, scoring='neg_mean_squared_error')
svm_scores = -cross_val_score(svm_regressor, X, y, cv=10, scoring='neg_mean_squared_error')

scores = [rf_scores, nn_scores, svm_scores]
labels = ['Random Forest', 'Neural Network', 'Support Vector Machine']

plt.boxplot(scores)
plt.xticks([1, 2, 3], labels)
plt.ylabel('Mean Squared Error')
plt.show()


#PART 2 – OPTIMISATION

import pandas as pd

def read_input_file('C:\Users\Joker\Downloads\containers.csv'):
    """
    Reads the input file and returns a DataFrame object.
    """
    df = pd.read_csv(file_path)
    return df

