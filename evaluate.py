import tensorflow as tf
form BERT_NER import *


estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=True,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=FLAGS.train_batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    predict_batch_size=FLAGS.predict_batch_size)s

predict_input_fn = file_based_input_fn_builder(
    input_file=predict_file,
    seq_length=64,
    is_training=False,
    drop_remainder=False)

result = estimator.predict(input_fn=predict_input_fn)
