import tensorflow as tf

IMAGE_SIZE = 28

def model(num_classes=10, batch_size = None):
  input_shape = (IMAGE_SIZE, IMAGE_SIZE,)
  img_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
  
  x = img_input
  
  x = tf.keras.layers.BatchNormalization, axis=3)(x)
  x = tf.keras.layers.Conv2D(filters=32, kernel_size=5, stride=1, padding='same')
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

  x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, stride=1, padding='same')
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, stride=2, padding='same')
  x = tf.keras.layers.Activation('relu')(x)
  
  x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, stride=1, padding='same')
  x = tf.keras.layers.Activation('relu')(x)
  
  x = tf.keras.layers.AveragePooling2D(7, strides=1, padding='same')(x)
  x = tf.keras.layers.Flatten(input_shape=(8,8))(x)
  x = tf.keras.layers.Dense(10)(x)
  
  x = tf.keras.layers.Softmax()(x)
  
  return tf.keras.Model(img_input, x, name='simpleNN')
