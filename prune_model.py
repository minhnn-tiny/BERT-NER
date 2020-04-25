import tensorflow as tf
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_checkpoint", None,
    "The input checkpoint file path"
)

flags.DEFINE_string(
    "output_file", None,
    "The output directory of the pruned model"
)


def main(argv):
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(
        FLAGS.input_checkpoint + '.meta')
    imported_meta.restore(sess, FLAGS.input_checkpoint)
    my_vars = []
    for var in tf.all_variables():
        if 'adam_v' not in var.name and 'adam_m' not in var.name:
            my_vars.append(var)
    saver = tf.train.Saver(my_vars)
    saver.save(sess, FLAGS.output_file)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_checkpoint")
    flags.mark_flag_as_required("output_file")

    app.run(main)
