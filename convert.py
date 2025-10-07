import tensorflowjs as tfjs
import tensorflow as tf

model = tf.keras.models.load_model("mongolian_script_model.h5")
tfjs.converters.save_keras_model(model, "tfjs_model")
