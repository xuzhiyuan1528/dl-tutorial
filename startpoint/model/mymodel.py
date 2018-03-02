import tensorflow as tf


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self._input_shape = [-1, 28, 28, 1]

        self.conv1 = tf.layers.Conv2D(
            32, 5, padding='same', activation=tf.nn.relu)
        self.conv2 = tf.layers.Conv2D(
            64, 5, padding='same', activation=tf.nn.relu)

        self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(10)

        self.dropout = tf.layers.Dropout(0.4)
        self.max_pool2d = tf.layers.MaxPooling2D(
            (2, 2), (2, 2), padding='same')

        def __call__(self, inputs, training):
            net = tf.reshape(inputs, self._input_shape)
            net = self.conv1(net)
            net = self.max_pool2d(net)
            net = self.conv2(net)
            net = self.max_pool2d(net)

            net = tf.layers.flatten(net)
            net = self.fc1(net)
            net = self.dropout(net, training=training)
            return self.fc2(net)
