import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell

import numpy as np


class LSTMAutoencoder(object):

    """Basic version of LSTM-autoencoder.
  (cf. http://arxiv.org/abs/1502.04681)
  Usage:
    ae = LSTMAutoencoder(hidden_num, inputs)
    sess.run(ae.train)
  """

    def __init__(
        self,
        hidden_num,
        inputs,
        cell=None,
        optimizer=None,
        reverse=True,
        decode_without_input=False,
        ):
        """
    Args:
      hidden_num : number of hidden elements of each LSTM unit.
      inputs : a list of input tensors with size
              (batch_num x elem_num)
      cell : an rnn cell object (the default option
            is `tf.python.ops.rnn_cell.LSTMCell`)
      optimizer : optimizer for rnn (the default option is
              `tf.train.AdamOptimizer`)
      reverse : Option to decode in reverse order.
      decode_without_input : Option to decode without input.
    """

        self.batch_num = inputs[0].get_shape().as_list()[0]
        self.elem_num = inputs[0].get_shape().as_list()[1]

        if cell is None:
            self._enc_cell = LSTMCell(hidden_num)
            self._dec_cell = LSTMCell(hidden_num)
        else:
            self._enc_cell = cell
            self._dec_cell = cell



        with tf.variable_scope('encoder'):
            (self.z_codes, self.enc_state) = tf.nn.static_rnn(self._enc_cell, inputs, dtype=tf.float32)

        with tf.variable_scope('decoder') as vs:
            dec_weight_ = tf.Variable(tf.truncated_normal([hidden_num,
                    self.elem_num], dtype=tf.float32), name='dec_weight'
                    )
            dec_bias_ = tf.Variable(tf.constant(0.1,
                                    shape=[self.elem_num],
                                    dtype=tf.float32), name='dec_bias')

            if decode_without_input:
                dec_inputs = [tf.zeros(tf.shape(inputs[0]),
                              dtype=tf.float32) for _ in
                              range(len(inputs))]
                (dec_outputs, dec_state) = tf.nn.static_rnn(self._dec_cell, dec_inputs, initial_state=self.enc_state,
                        dtype=tf.float32)
                if reverse:
                    dec_outputs = dec_outputs[::-1]
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0,
                        2])
                dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0),
                        [self.batch_num, 1, 1])
                self.output_ = tf.matmul(dec_output_, dec_weight_) + dec_bias_
            else:

                dec_state = self.enc_state
                dec_input_ = tf.zeros(tf.shape(inputs[0]),
                        dtype=tf.float32)
                dec_outputs = []
                for step in range(len(inputs)):
                    if step > 0:
                        vs.reuse_variables()
                    (dec_input_, dec_state) = \
                        self._dec_cell(dec_input_, dec_state)
                    dec_input_ = tf.matmul(dec_input_, dec_weight_) \
                        + dec_bias_
                    dec_outputs.append(dec_input_)
                if reverse:
                    dec_outputs = dec_outputs[::-1]
                self.output_ = tf.transpose(tf.stack(dec_outputs), [1,
                        0, 2])

        self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2])
        self.loss = tf.reduce_mean(tf.square(self.input_
                                   - self.output_))

        if optimizer is None:
            self.train = tf.train.AdamOptimizer().minimize(self.loss)
        else:
            self.train = optimizer.minimize(self.loss)


tf.reset_default_graph()
tf.set_random_seed(2016)
np.random.seed(2016)


# Constants
batch_num = 128
hidden_num = 12
step_num = 8
elem_num = 1
iteration = 10000

# placeholder list
p_input = tf.placeholder(tf.float32, shape=(batch_num, step_num, elem_num))
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(iteration):
        """Random sequences.
          Every sequence has size batch_num * step_num * elem_num 
          Each step number increases 1 by 1.
          An initial number of each sequence is in the range from 0 to 19.
          (ex. [8. 9. 10. 11. 12. 13. 14. 15])
        """
        r = np.random.randint(20, size=batch_num).reshape([batch_num, 1, 1])
        r = np.tile(r, (1, step_num, elem_num))
        d = np.linspace(0, step_num, step_num, endpoint=False).reshape([1, step_num, elem_num])
        d = np.tile(d, (batch_num, 1, 1))
        random_sequences = r + d

        (loss_val, _) = sess.run([ae.loss, ae.train], {p_input: random_sequences})
        print('iter %d:' % (i + 1), loss_val)

    (input_, output_) = sess.run([ae.input_, ae.output_], {p_input: r + d})
    print('train result :')
    print('input :', input_[0, :, :].flatten())
    print('output :', output_[0, :, :].flatten())