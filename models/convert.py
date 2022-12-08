import tensorflow as tf

saved_model_dir = 'C:\\Users\\Bavo Lesy\\PycharmProjects\\RaceAI\\models\\saved_model_v1'
model = tf.saved_model.load(saved_model_dir)
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, 600, 800, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# converter = tf.lite.TFLiteConverter.from_saved_model(
#     saved_model_dir, signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

fo = open(
   'C:\\Users\\Bavo Lesy\\PycharmProjects\\RaceAI\\models\\saved_model_v1_tflite\\model.tflite', 'wb')
fo.write(tflite_model)
fo.close()

