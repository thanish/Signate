import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from glob import glob
import shutil
from boruta import BorutaPy
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures


# Downloading the required files

# print('Beginning submission file download...')

# submission_file_url = 'https://signate-prd.s3.ap-northeast-1.amazonaws.com/datasets/155/sample_submit.csv?X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIOPRBJGKMEQPRJUA%2F20190903%2Fap-northeast-1%2Fs3%2Faws4_request&X-Amz-Date=20190903T043234Z&X-Amz-SignedHeaders=host&X-Amz-Expires=600&X-Amz-Signature=1bbf4dc3209139e3a8efd12e75277a2db8f91fbf5b72975b6969d13d2df45f5f'
# filename = os.getcwd() + '/submission_file.csv'
# urllib.request.urlretrieve(submission_file_url, filename)

# print('Beginning train file download...')
# train_url = 'https://signate-prd.s3.ap-northeast-1.amazonaws.com/datasets/155/train.csv?X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIOPRBJGKMEQPRJUA%2F20190916%2Fap-northeast-1%2Fs3%2Faws4_request&X-Amz-Date=20190916T092028Z&X-Amz-SignedHeaders=host&X-Amz-Expires=600&X-Amz-Signature=005b431edf7913d6607f9257a59cfad540028eb8104203507cd307be3d598b84'
# filename = os.getcwd() + '/train_1.csv'
# urllib.request.urlretrieve(train_url, filename)

# print('Beginning test file download...')
# test_url = 'https://signate-prd.s3.ap-northeast-1.amazonaws.com/datasets/155/test.csv?X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIOPRBJGKMEQPRJUA%2F20190903%2Fap-northeast-1%2Fs3%2Faws4_request&X-Amz-Date=20190903T043239Z&X-Amz-SignedHeaders=host&X-Amz-Expires=600&X-Amz-Signature=763d4722f7d504ae5f344f8bb019b998262be760b5e6a403c9dc6dad21ac07d0'
# filename = os.getcwd() + '/test.csv'
# urllib.request.urlretrieve(test_url, filename)

# print("Download complete")

#Reading the files
train_prod = pd.read_csv('train.csv')
test_prod = pd.read_csv('test.csv')

# Filtering
print("Before filter", train_prod.shape)
# train_prod = train_prod.loc[train_prod.Score != -1, :]
print("After filter", train_prod.shape)

# Features with zero variance
zer_var_col = train_prod.columns[train_prod.var() == 0] 
low_var = train_prod.columns[train_prod.var() <= 0.1]

print("Total Zero variance columns ", len(zer_var_col))
print("Total Low variance columns ", len(low_var))

corr_matrix = train_prod[train_prod.columns.difference(['Score', 'ID'] + zer_var_col.tolist() + low_var.tolist())].corr() 
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
corr_columns_to_drop = [column for column in upper.columns if any(upper[column] >= 0.96)]
len(corr_columns_to_drop)


permutation_col_gr_than_eq_0 = ['col3005', 'col938', 'col3754', 'col1487', 'col1046', 'col350', 'col2947', 'col1335',
                               'col2615', 'col1048', 'col3305', 'col2452', 'col1938', 'col3482', 'col1102', 'col1929',
                               'col280', 'col3336', 'col2433', 'col2818', 'col3246', 'col1419', 'col657', 'col21',
                               'col598', 'col2531', 'col2936', 'col2977', 'col604', 'col2899', 'col2894', 'col2366',
                               'col3694', 'col3769', 'col3616', 'col2015', 'col1074', 'col1001', 'col1342', 'col1572',
                               'col2221', 'col219', 'col2663', 'col445', 'col85', 'col2853', 'col2711', 'col766',
                               'col1637', 'col2954', 'col1883', 'col48', 'col1506', 'col1838', 'col398', 'col1138',
                               'col901', 'col2875', 'col69', 'col2807', 'col2373', 'col1862', 'col2811', 'col129',
                               'col2024', 'col1223', 'col638', 'col110', 'col2486', 'col2278', 'col1954', 'col1219',
                               'col3805', 'col2939', 'col1264', 'col1599', 'col2840', 'col1051', 'col524', 'col3118',
                               'col3311', 'col2336', 'col1359', 'col690', 'col2413', 'col3344', 'col3575', 'col1316',
                               'col279', 'col763', 'col1547', 'col3107', 'col3520', 'col1632', 'col2111', 'col3548',
                               'col3308', 'col2372', 'col1857', 'col1318', 'col961', 'col2869', 'col1273', 'col389',
                               'col1675', 'col754', 'col319', 'col3284', 'col1488', 'col115', 'col1849', 'col3046',
                               'col1894', 'col1209', 'col1452', 'col1947', 'col438', 'col2395', 'col1897', 'col2101',
                               'col1232', 'col1489', 'col1542', 'col3628', 'col2677', 'col1696', 'col3065', 'col239',
                               'col625', 'col1816', 'col2120', 'col220', 'col873', 'col1781', 'col1015', 'col2732',
                               'col1612', 'col2169', 'col3488', 'col957', 'col496', 'col1236', 'col1060', 'col633',
                               'col1988', 'col2898', 'col1181', 'col1918', 'col1371', 'col342', 'col3180']

dep = 'Score'
cols_2_drop = ['ID']
final_drop = [dep] + cols_2_drop + zer_var_col.tolist() + low_var.tolist() + corr_columns_to_drop + permutation_col_gr_than_eq_0 #+ other_manual_columns #+ corr_2_score_drop.index.tolist() #+ LGB_imp.loc[LGB_imp.Value == 0,].Feature.tolist()
indep = train_prod.columns.difference(final_drop)

# Splitting data to train and validation
np.random.seed(100)
train_local_X, test_local_X, train_local_Y, test_local_Y = train_test_split(train_prod[indep],
                                                                            train_prod[dep], 
                                                                            test_size = 0.3)
print(train_local_X.shape, test_local_X.shape, train_local_Y.shape, test_local_Y.shape)

# Random Forest
# On local
np.random.seed(100)
RF = RandomForestRegressor(n_jobs = -1, n_estimators = 500)
RF.fit(train_local_X, train_local_Y)

RF_local_prediction = RF.predict(test_local_X)
r2_score(RF_local_prediction, test_local_Y)

# Use Bortua to get important features
boruta_selector = BorutaPy(RF, n_estimators = "auto", verbose=2)

x = train_local_X.values
y = train_local_Y.values
boruta_selector.fit(x,y)


print("==============BORUTA==============")
print (boruta_selector.n_features_)

feature_df = pd.DataFrame(train_prod[indep].columns.tolist(), columns=['features'])
feature_df['rank_order'] = boruta_selector.ranking_
feature_df = feature_df.sort_values('rank_order', ascending=True).reset_index(drop=True)

print ('\n Top %d features:' % boruta_selector.n_features_)
print (feature_df.head(boruta_selector.n_features_))

feature_df.to_csv('Boruta_feature_df.csv', index = False)

feature_df = pd.read_csv('Boruta_feature_df.csv')

boruta_features = feature_df.features[feature_df.rank_order.isin(np.arange(1000))].tolist()
print(len(boruta_features))

indep = np.array(boruta_features)
np.random.seed(100)
train_local_X, test_local_X, train_local_Y, test_local_Y = train_test_split(train_prod[indep],
                                                                            train_prod[dep], 
                                                                            test_size = 0.3)

print(train_local_X.shape, test_local_X.shape, train_local_Y.shape, test_local_Y.shape)

# XGBOOST
def permutation_importance(X, y_actual, model): 
    perm = {}
    dtest_local = xgb.DMatrix(data = X, label = y_actual)
    y_baseline_prediction = model.predict(dtest_local)
    baseline = r2_score(y_pred = y_baseline_prediction, y_true = y_actual)
    
    for index, cols in enumerate(X.columns):
        value = X[cols].copy()
        print(index, cols)
        X[cols] = np.random.permutation(X[cols].values)
        dtest_local = xgb.DMatrix(data = X, label = y_actual)
        y_permuted_prediction = model.predict(dtest_local)
        perm[cols] = r2_score(y_pred = y_permuted_prediction, y_true = y_actual) - baseline
        X[cols] = value
    perm_imp_DF = pd.DataFrame({'col_name': list(perm.keys()),
                                'permutation_value' : list(perm.values())})
    return perm_imp_DF

def custom_r2(preds, dtrain):
    labels = dtrain.get_label()
    R2 = r2_score(y_pred = preds, y_true = labels)
    return 'R2', R2


dtrain_prod = xgb.DMatrix(data = train_prod[indep], label = train_prod[dep])
dtest_prod = xgb.DMatrix(data = test_prod[indep])
dtrain_local = xgb.DMatrix(data = train_local_X, label = train_local_Y)
dtest_local = xgb.DMatrix(data = test_local_X, label = test_local_Y)

eval_set = [(dtrain_local,'train'), (dtest_local,'test')]

# Cross validation

num_rounds = 10000
params = {'objective' : 'reg:squarederror',
          'eval_metric': 'rmse',
          'max_depth' : 6,
          'eta' : 0.006,
          'subsample': 0.6,
          'colsample_bytree': 0.6
          ,'tree_method' : 'gpu_hist'
          }
xgb_model_cv = xgb.cv(params,
                      dtrain_prod,
                      nfold = 5,
                      #evals = eval_set,
                      num_boost_round = num_rounds,
                      #feval = custom_r2,
                      #maximize = True,
                      verbose_eval = True,
                      early_stopping_rounds = 30)


cv_best_iter = xgb_model_cv['test-rmse-mean'].index[xgb_model_cv['test-rmse-mean'] == xgb_model_cv['test-rmse-mean'].min()][0]
print("CV best iter:",cv_best_iter, "Best error:", xgb_model_cv['test-rmse-mean'].min())

# CV best iter: 9999 Best error: 0.5705560000000001

# Find the important features using permutation_importance
# Permutation importance
imp_DF = permutation_importance(X = test_local_X, y_actual = test_local_Y, model = xgb_model)
len(imp_DF.col_name[imp_DF.permutation_value > 0])


# Prod

xgb_model_prod = xgb.train(params,
                           dtrain_prod,
                           evals = eval_set,
                           #num_boost_round = (xgb_model.best_iteration + int(xgb_model.best_iteration * 0.60)),
                           num_boost_round = (cv_best_iter + int(cv_best_iter * 0.75)),
                           feval = custom_r2,
                           #verbose_eval = True,
                           #early_stopping_rounds = 20
                          )

xgb_prod_pred = xgb_model_prod.predict(dtest_prod)

XGB_sub = pd.DataFrame({'ID' : test_prod.ID,
                        'pred' : xgb_prod_pred})
XGB_sub.to_csv('XGB_sub_26.csv', header = None, index = False)


# Light GBM
lg = lgb.LGBMRegressor(silent = False)
lgb_train_prod = lgb.Dataset(train_prod[indep], train_prod[dep], free_raw_data=False )
lgb_train_local = lgb.Dataset(train_local_X, train_local_Y,free_raw_data=False )
lgb_test_local = lgb.Dataset(test_local_X, test_local_Y, free_raw_data=False, reference = lgb_train_local)

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    R2 = r2_score(y_pred = preds, y_true = labels)
    return 'R2', R2, True

params = {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        #'metric': {'rmse'},
        'max_depth': 6,
        'num_leaves': 32,
        'learning_rate': 0.004,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 1,
        'verbose': 1
        }

# 5 Fold CV
cat_column = []
for col in train_prod[indep].columns:
    if (len(np.unique(train_prod[col])) <= 4):
        cat_column.append(col)
print("Total cat columns", len(cat_column))

nrounds = 100000
folds = 5
#LGBM cross validation
np.random.seed(100)
lgbm_prod_cv= lgb.cv(params, 
                     lgb_train_prod, 
                     nfold = folds ,
                     num_boost_round = nrounds,
                     verbose_eval = 1,
                     feval = evalerror,
                     early_stopping_rounds = 50,
                     stratified=False , categorical_feature = cat_column
                     )

error_array = np.array(lgbm_prod_cv['R2-mean'])
best_error = error_array.max()
best_round = np.where(error_array == best_error)[0][0]

print("")
print("The best error", best_error)
print("The best iteration" , best_round )

cv_best_round = int(round(best_round/(1 - (1/folds))))
print("Final_round", cv_best_round )


cat_column = []
for col in train_prod[indep].columns:
    if (len(np.unique(train_prod[col])) <= 4):
        cat_column.append(col)
print("Total cat columns", len(cat_column))

params = {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        #'metric': {'rmse'},
        'max_depth': 6,
        'num_leaves': 32,
        'learning_rate': 0.01,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 1,
        'verbose': 1
}


#LGBM final model
np.random.seed(100)
lgbm_model = lgb.train(params, 
                       lgb_train_local,
                       valid_sets = lgb_test_local,
                       feval = evalerror,
                       early_stopping_rounds = 50,
                       num_boost_round = nrounds , categorical_feature = cat_column
                      )
pd.DataFrame(sorted(zip(lgbm_model.feature_importance(), indep)), 
                               columns=['Value','Feature']).sort_values(['Value'], ascending = False)

# [5811]	valid_0's l2: 0.326002	valid_0's R2: 0.622792 - 'feature_fraction': 0.5, bagging_fraction - 0.

# Prod
np.random.seed(100)
lgbm_prod_model = lgb.train(params, 
                            lgb_train_prod,
                            valid_sets = lgb_test_local,
                            feval = evalerror,
                            early_stopping_rounds = 10,
                            #num_boost_round = lgbm_model.best_iteration + int(lgbm_model.best_iteration * 0.55)
                            num_boost_round = best_round + int(best_round * 0.55)
                            )
lgbm_prod_predict = lgbm_prod_model.predict(test_prod[indep])

LGB_model_sub = pd.DataFrame({'ID':test_prod.ID,
                              'pred':lgbm_prod_predict})
LGB_model_sub.to_csv('sub_LGB_18.csv',header = None, index = False)


