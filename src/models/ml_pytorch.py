import pandas as pd
import numpy as np
import tensorflow as tf
import featEng as fe
from sklearn.metrics import mean_squared_error
from keras import backend as K

def get_tf_adam_mse_3l_10_20_1_relu():
  tf.keras.backend.set_floatx('float64')
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
  ])

  model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_squared_error','MeanAbsolutePercentageError'])
  return model


def get_tf_adam_mse_5l_1024_512_256_128_1_relu():
  tf.keras.backend.set_floatx('float64')
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

class ModelProcessorTF():
  model = get_tf_adam_mse_3l_10_20_1_relu()
  X_train = []
  y_train = []
  X_test = []
  y_test= []
  df_test_x_y = []

  def apply_model(self, model_param=None):
    if model_param is not None:
      self.model = model_param

    df_test_x_y = []
    print("Fit model on training data")
    print (self.X_train)
    print (self.y_train)
    history = self.model.fit(
        self.X_train,
        self.y_train,
        epochs=500
    )
    print("Evaluate on test data")
    results = self.model.evaluate(self.X_test, self.y_test, batch_size=128)
    print("test loss, test acc:", results)

    predictions = self.model.predict(self.X_test)

    print(np.sqrt(mean_squared_error(self.y_test,predictions)))

    self.df_test_x_y = pd.concat([self.X_test, self.y_test], axis=1)

    my_list = map(lambda x: x[0], predictions)
    predictions = pd.Series(my_list)

    self.df_test_x_y['predictions'] = predictions.values
    self.df_test_x_y['error_perc'] = (self.df_test_x_y['predictions']-self.df_test_x_y['future_price'])/self.df_test_x_y['future_price']

    self.df_test_x_y.to_csv("predictions.csv")

  def __init__(self, df):
    K.clear_session()
    self.X_train, self.X_test, self.y_train, self.y_test = fe.splittingForTraining(df)
    return self.apply_model()
