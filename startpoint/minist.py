import tensorflow as tf
import sys

from startpoint import dataset
from startpoint.model.mymodel import MyModel
from startpoint.parser.arg_parser import MyArgParser

def validate_batch_size(batch_size):
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    if not num_gpus:
        raise ValueError('no GPU found')

    remainder = batch_size % num_gpus
    if remainder:
        err = ('When running with multiple GPUs, batch size '
               'must be a multiple of the number of available GPUs. '
               'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
               ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)


def model_fn(features, labels, mode, params):
    image = features
    model = MyModel(image, None)
    if isinstance(image, dict):
        image = features['image']

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(image, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.PREDICT,
            predictions = predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            }
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(1e-4)

        if params.get('multi_gpu'):
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        logits = model(image, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1))

        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(
                    labels=labels,
                    predictions=tf.argmax(logits, axis=1))
            })


def main(unused_argv):
    model_function = model_fn

    if FLAGS.multi_gpu:
        validate_batch_size(FLAGS.batch_size)

        model_function = tf.contrib.estimator.replicate_model_fn(
            model_fn, loss_reduction=tf.losses.Reduction.MEAN)

    mnist_classifier = tf.estimator.Estimator(
        model_fn = model_function,
        model_dir = FLAGS.mod_dir)

    def train_input_fn():
        ds = dataset.train(FLAGS.data_dir)
        ds = ds.cache().shuffle(buffer_size=50000).batch(FLAGS.batch_size).repeat(
            FLAGS.train_epochs)
        return ds

    tensors_to_log = {'train_accuracy': 'train_accuracy'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    mnist_classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

    def eval_input_fn():
        return dataset.test(FLAGS.data_dir).batch(
            FLAGS.batch_size).make_one_shot_iterator().get_next()

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print('Evaluation Result: {0}'.format(eval_results))

    if FLAGS.mod_dir is not None:
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': image
        })
        mnist_classifier.export_savedmodel(FLAGS.mod_dir, input_fn)


if __name__ == '__main__':
    parser = MyArgParser()
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main, argv=[sys.argv[0]] + unparsed)
