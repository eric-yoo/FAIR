import tensorflow as tf

IMAGE_SIZE = 28

def model(num_classes=10, batch_size = None):
  input_shape = (IMAGE_SIZE, IMAGE_SIZE,)
  img_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
  
  x = img_input

  x = tf.keras.layers.Flatten(input_shape=(28, 28))(x)
  x = tf.keras.layers.Dense(512, activation='relu')(x)
  x = tf.keras.layers.Dense(512, activation='relu')(x)
  x = tf.keras.layers.Dense(512, activation='relu')(x)
  x = tf.keras.layers.Dense(10)(x)
  x = tf.keras.layers.Softmax()(x)
  
  return tf.keras.Model(img_input, x, name='simpleNN')
