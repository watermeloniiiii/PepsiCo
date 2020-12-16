
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pickle

global itr
itr = 100

font1 = {
    'family': 'Arial',
    'size':20
}
np.set_printoptions(suppress=True)

## training data
results = pd.read_csv(r'.\combined.csv')




ind_var = ['evi', 'ndvi', 'gdvi', 'osavi', 'psri', 'ndre'];
d_var = ['TILLERS_SQ'];

x = np.array(results[ind_var]);
y = np.array(results[d_var]).ravel();
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# ################
# ##### parameter tuning using gridsearchCV #####
# ################
# param_test1 = {'n_estimators': range(5, 101, 5),
#                'min_samples_split': range(2, 11, 1),
#                'min_samples_leaf': range(1, 11, 1)}
# gsearch1 = GridSearchCV(estimator=RandomForestRegressor(max_depth=None, max_features=None, random_state=10),
#                        param_grid = param_test1, scoring='r2', cv=10)
#
# gsearch1.fit(x_train, y_train)


regr1 = RandomForestRegressor(n_estimators=80,
                              min_samples_split=2,
                              min_samples_leaf=1,
                              max_depth=None,
                              max_features=None,
                              random_state=9)
regr1.fit(x_train, y_train)
filename = 'finalized_model.sav'
pickle.dump(regr1, open(filename, 'wb'))
prdct_train = regr1.predict(x_train)
prdct_test = regr1.predict(x_test)
prdct_y = regr1.predict(x)
R2_train = np.corrcoef(prdct_train, y_train)[0,1] ** 2
R2_test = np.corrcoef(prdct_test, y_test)[0,1] ** 2
RMSE_train = np.sqrt(np.mean((prdct_train - y_train)**2))
RMSE_test = np.sqrt(np.mean((prdct_test - y_test)**2))
print (R2_train)
print (R2_test)
print (RMSE_train)
print (RMSE_test)







