# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:53:16 2018

@author: adithya
"""
#importing libraries and dataset
import numpy as np
import pandas as pd
xls = pd.ExcelFile('Demandv1.1.xlsx')
df1 = pd.read_excel(xls, 'HeadCount')
df2 = pd.read_excel(xls, 'DemandTrend')

#renaming columns
df1.columns = df1.columns.str.replace(' ', '_')
df2.columns = df2.columns.str.replace(' ', '_')

#preparing for encoding
df2 = df2.groupby(['Month_DD_Raised','SkillList','Location','Experience_Grade','Skill_Group','Demand_Source','Practice'], as_index=False)['No._of_FTE_Request_Raised'].sum()
#sorting acooring to month
df2.Month_DD_Raised = pd.Categorical(df2.Month_DD_Raised,categories=['January','February','March','May','June','August','October','December'],
                                    ordered=True)
df2.sort_values('Month_DD_Raised', inplace=True)

#splitting dataset in categorical and numerical for encoding
c_df2 = df2.select_dtypes(exclude='int64')
n_df2 = df2.select_dtypes(include='int64')

# label encoding for categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
c_df2 = c_df2.apply(le.fit_transform)

#combining encoded categorical and numerical data
df2 = pd.concat([c_df2,n_df2],axis=1)

#checking for correlation of the target variable with other variables
correlations = df2.corr()
correlations = correlations['No._of_FTE_Request_Raised'].sort_values(ascending=False)
features = correlations.index[0:7]
correlations

#dropping non correlated variables for better prediction
df2.drop(['Skill_Group','Experience_Grade'],axis=1,inplace=True)

#preparing data for training
X = df2.iloc[:,1:].values
y = df2.iloc[:,0].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#checking for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#importing regressors for training
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
#importing accuracy metrics, model selection 
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Fitting Random Forest Regressor to the Training set
rf = RandomForestRegressor()
param_rf= {"n_estimators":[100],"criterion":["mse","mae"],"min_samples_split" : [2, 3, 5, 10], 
                 "max_features" : ["auto", "log2"]}
grid_rf = GridSearchCV(rf, param_rf, verbose=1, scoring="r2")
grid_rf.fit(X_train, y_train)

rf = grid_rf.best_estimator_
rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)
r2_rf = r2_score(y_test,rf_pred)
rmse_rf=np.sqrt(mean_squared_error(y_test,rf_pred))
print('r2 score:' + str(r2_rf))
print('Rmse score:'+ str(rmse_rf))

# Fitting xgboost to the Training set
xgb = XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=1000, silent=False, 
                   objective='reg:linear', booster='gbtree', n_jobs=1, nthread=8,
                   gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
                   colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
                   scale_pos_weight=1, base_score=0.5, random_state=0, seed=None,)

xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)
r2_xgb=r2_score(y_test,xgb_pred)
rmse_xgb=np.sqrt(mean_squared_error(y_test,xgb_pred))
print('best r2 score:'+ str(r2_xgb))
print('best rmse score:'+ str(rmse_xgb))

import numpy as np
import pickle
file = 'ixi_model.pkl'
pickle.dump(rf, open(file,'wb'))


