import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
  df_test_x_y = []
  X_train = []
  X_test = []
  y_train = []
  y_test = []
  LOOK_WINDOW = -365

  def apply_model(self, model_param=None):
    if model_param is not None:
      self.model = model_param

    df_test_x_y = []
    print("Fit model on training data")
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

    #df_test_x_y.to_csv("predictions.csv")

  def __init__(self, df):
    df['future_price'] = df.loc[:, 'Close'].shift(self.LOOK_WINDOW)

    #Drop the n amount of rows, as they are empty
    df.drop(df.tail(abs(self.LOOK_WINDOW)).index,inplace=True)

    target = df.pop('future_price')
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df,
                                                    target,
                                                    test_size=0.33,
                                                    random_state=42)
    return self.apply_model()
