# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
# Import supplementary visualizations code visuals.py
import visuals as vs
# Pretty display for notebooks
%matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

#Calculate Statistics
# Minimum price of the data
minimum_price = np.min(prices)
# Maximum price of the data
maximum_price = np.max(prices)
#Mean price of the data
mean_price = np.mean(prices)
#Median price of the data
median_price = np.median(prices)
#Standard deviation of prices of the data
std_price = np.std(prices)
# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))

# Define performence metrics
#Import 'r2_score'
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):#Calculates and returns the performance score between 
       									 #true and predicted values based on the metric chosen.    
   
    score = r2_score (y_true, y_predict)    
    return score
#Shuffle and split the data into training and testing subsets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,prices,test_size=0.2,random_state=18) 

# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)
vs.ModelComplexity(X_train, y_train)

 #Performs grid search over the 'max_depth' parameter for a 
 #decision tree regressor trained on the input data [X, y].
 #Fitting a Model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV


def fit_model(X, y):    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()
    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':list(range(1,11))} 
    #Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)
    #Create the grid search object
    grid = GridSearchCV(regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)
    # Return the optimal model after fitting the data
    return grid.best_estimator_

#Optimal Model
# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)
# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

#Predicting Selling Prices
# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3
# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))
# Sensitivity
 vs.PredictTrials(features, prices, fit_model, client_data)
