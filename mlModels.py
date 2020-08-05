import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import gc
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

def get_tf_adam_mse_3l_10_20_1_relu(training_data=None):
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
  ])

  model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_squared_error','MeanAbsolutePercentageError'])
  return model

def get_tf_adam_mse_100_1_LSTM(training_data=None):
  model = Sequential()
  print("training_data.shape[0]")
  print(training_data.shape[0])
  print("training_data.shape[1]")
  print(training_data.shape[1])
  model.add(LSTM(100, input_shape = (1, training_data.shape[1])))
  model.add(Dense(1))
  model.compile(optimizer = 'adam', loss='mse')
  return model

def get_tf_RMSprop_mse_3l_10_20_1_relu(training_data=None):
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
  ])

  opt = tf.keras.optimizers.RMSprop()

  model.compile(optimizer=opt,
                loss='mean_squared_error',
                metrics=['mean_squared_error','MeanAbsolutePercentageError'])
  return model


def get_tf_adam_mse_5l_1024_512_256_128_1_relu(training_data=None):
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
  ])

  model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_squared_error','MeanAbsolutePercentageError'])
  return model

#class ModelProcessorTF():
class ModelHandler():

  def apply_tf_model(self, model_param=None):
    if model_param != None:
      self.tf_model = model_param

    df_tf_output = []
    tf.keras.backend.set_floatx('float64')
    print("Fit model on training data")

    #X_train = self.X_train.reshape(-1, 1, self.X_train.shape[1])
    #X_test = self.X_test.reshape(-1, 1, self.X_test.shape[1])
    #y_train = self.y_train.reshape(-1, 1, self.y_train.shape[1])
    #y_test = self.y_test.reshape(-1, 1, self.y_test.shape[1])

    history = self.tf_model.fit(
        self.X_train,
        self.y_train,
        epochs=50,
        #batch_size=128,
        validation_data=(self.X_test, self.y_test),
    )
    print("Evaluate on test data")
    results = self.tf_model.evaluate(self.X_test, self.y_test, batch_size=128)
    print("test loss, test acc:", results)

    predictions = self.tf_model.predict(self.X_test)

    print(np.sqrt(mean_squared_error(self.y_test,predictions)))

    self.df_tf_output = pd.concat([self.X_test, self.y_test], axis=1)

    my_list = map(lambda x: x[0], predictions)
    predictions = pd.Series(my_list)

    self.df_tf_output['predictions'] = predictions.values

    tf.keras.backend.clear_session()
    del self.tf_model, history, predictions
    gc.collect()

    #self.df_test_x_y.to_csv("predictions.csv")

  def apply_randomforest_model(self):
    self.randomforest_model = RandomForestRegressor()
    self.randomforest_model.fit(self.X_train, self.y_train)
    rf_predictions = self.randomforest_model.predict(self.X_test)
    print(np.sqrt(mean_squared_error(self.y_test,rf_predictions)))
    self.df_randomforest_output = pd.concat([self.X_test, self.y_test], axis=1)

    rf_predictions = pd.Series(rf_predictions)

    self.df_randomforest_output['predictions'] = rf_predictions.values

  def apply_lasso_model(self):
    self.lasso_model = LassoCV()
    self.lasso_model.fit(self.X_train, self.y_train)
    lasso_predictions = self.lasso_model.predict(self.X_test)
    print(np.sqrt(mean_squared_error(self.y_test,lasso_predictions)))

    self.df_lasso_output = pd.concat([self.X_test, self.y_test], axis=1)

    lasso_predictions = pd.Series(lasso_predictions)

    self.df_lasso_output['predictions'] = lasso_predictions.values

  def __init__(self, df):
    self.df_tf_output = []
    self.df_svm_output = []
    self.X_train = []
    self.X_test = []
    self.y_train = []
    self.y_test = []
    target = df.pop('future_price')

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df,
                                                    target,
                                                    test_size=0.33,
                                                    random_state=42)
    self.tf_model = get_tf_adam_mse_3l_10_20_1_relu()
    self.apply_tf_model()
    self.apply_randomforest_model()
    self.apply_lasso_model()

    return None
