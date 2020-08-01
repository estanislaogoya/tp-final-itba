import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def get_compiled_model():
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

def fitDataFrame(df):
    look_window = -365
    df['future_price'] = df.loc[:, 'Close'].shift(look_window)

    #Drop the n amount of rows, as they are empty
    df.drop(df.tail(abs(look_window)).index,inplace=True)

    target = df.pop('future_price')
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                    target,
                                                    test_size=0.33,
                                                    random_state=42)
    #dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    #for feat, targ in dataset.take(1):
        #print ('Features: {}, Target: {}'.format(feat, targ))
    #train_dataset = dataset.shuffle(len(X_train)).batch(1)
    
    model = get_compiled_model()
    print("Fit model on training data")
    history = model.fit(
        X_train,
        y_train,
        epochs=15
    )

    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test acc:", results)

    predictions = model.predict(X_test)

    print(np.sqrt(mean_squared_error(y_test,predictions)))

    df_test_x_y = pd.concat([X_test, y_test], axis=1)

    print('datos de X_test')
    print(len(X_test))
    print(X_test)

    print('datos de y_test')
    print(len(y_test))
    print(type(y_test))
    print(y_test)

    my_list = map(lambda x: x[0], predictions)
    predictions = pd.Series(my_list)
    print('datos de predictions')
    print(len(predictions))
    print('type of this predictions is:')
    print(type(predictions))
    print('raw values are:')
    print(predictions)

    #df_test_x_y.reset_index(drop=True, inplace=True)
    df_test_x_y['predictions'] = predictions
    
    #df_test_x_y = pd.concat([df_test_x_y, predictions], axis = 1, ignore_index=True)
 
    #DF_FULL = pd.concat([X_test, y_test, predictions], axis=1, sort=False)
    df_test_x_y.to_csv("predictions.csv")

    #print(history.history)
