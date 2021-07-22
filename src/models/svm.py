import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import mean_squared_error
import src.features.featEng as fe


def runSVM_SVR(df):
    X_train , X_test , y_train, y_test = fe.splittingForTraining(df)
    regr = svm.SVR()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    result_df = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
    result_df['diff_pred_val'] = result_df['y_test']-result_df['y_pred']
    result_df['diff_pred_pct'] = (result_df['y_test']-result_df['y_pred'])/result_df['y_test']
    return result_df
