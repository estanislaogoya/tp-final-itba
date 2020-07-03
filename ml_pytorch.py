import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_squared_error','MeanAbsolutePercentageError'])
  return model

def fitDataFrame(df):
    look_window = -365
    df['future_price'] = df['Close'].shift(look_window)

    #Drop the n amount of rows, as they are empty
    df.drop(df.tail(look_window).index,inplace=True)

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
        batch_size=64,
        epochs=15,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_test, y_test),
    )

    predictions = model.predict(X_test)

    df_test_x_y = pd.concat([X_test, y_test], axis=1)
    df_test_x_y.to_csv("df_test_x_y.csv")

    df2 = pd.DataFrame(predictions)
    DF_FULL = pd.concat([df_test_x_y, df2], axis = 1, ignore_index=True)
 
    #DF_FULL = pd.concat([X_test, y_test, predictions], axis=1, sort=False)
    DF_FULL.to_csv("predictions.csv")

    #print(history.history)
