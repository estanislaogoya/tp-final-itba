import tensorflow as tf

look_window = -365

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_squared_error'])
  return model

def fitDataFrame(df):
    df['future_price'] = df['Close'].shift(look_window)
    #Drop the n amount of rows, as they are empty
    df.drop(df.tail(look_window).index,inplace=True)
    target = df.pop('future_price')
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    for feat, targ in dataset.take(1):
        print ('Features: {}, Target: {}'.format(feat, targ))
    train_dataset = dataset.shuffle(len(df)).batch(1)
    model = get_compiled_model()
    model.fit(train_dataset, epochs=15)


