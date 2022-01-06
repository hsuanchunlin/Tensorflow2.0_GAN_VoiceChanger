import tensorflow as tf

#Basic block for 1D
class BasicBlock_1D(tf.keras.layers.Layer):

    def __init__(self, filter_num=1024, kernel_size=3, stride=1):
        super(BasicBlock_1D, self).__init__()
        #Set up layers
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num,
                                            kernel_size=3,
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.LayerNormalization(epsilon = 1e-06)
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_num,
                                            kernel_size=3,
                                            strides=1,
                                            padding="same")
        self.conv3 = tf.keras.layers.Conv1D(filters=filter_num//2,
                                            kernel_size=3,
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.LayerNormalization(epsilon = 1e-06)
        self.bn3 = tf.keras.layers.LayerNormalization(epsilon = 1e-06)


    def call(self, inputs, training=None, **kwargs):
        x1 = self.conv1(inputs)
        x1 = self.bn1(x1)
        x2 = self.conv2(inputs)
        x2 = self.bn2(x2)
        #This is the gated layer
        x = tf.multiply(x1, tf.sigmoid(x2))
        x = self.conv3(x)
        x = self.bn3(x)
        #add residual and x
        output = tf.keras.layers.add([inputs, x])

        return output

# Downsample 1D
class DownSample_1D(tf.keras.layers.Layer):
    def __init__(self, filter_num=1024, kernel_size=3, stride=1):
        super(DownSample_1D, self).__init__()
        #Set up layers
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num,
                                        kernel_size=3,
                                        strides=stride,
                                        padding="same")
        self.bn1 = tf.keras.layers.LayerNormalization(epsilon = 1e-06)
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_num,
                                        kernel_size=3,
                                        strides=stride,
                                        padding="same")
        self.bn2 = tf.keras.layers.LayerNormalization(epsilon = 1e-06)

    def call(self, inputs, training=None, **kwargs):
        x1 = self.conv1(inputs)
        x1 = self.bn1(x1)

        x2 = self.conv2(inputs)
        x2 = self.bn2(x2)

        output = tf.multiply(x1, tf.sigmoid(x2))

        return output

class DownSample_2D(tf.keras.layers.Layer):

    def __init__(self, filter_num=1024, kernel_size= (3,3), stride=1):
        super(DownSample_2D, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=(3, 3),
                                        strides=stride,
                                        padding="same")
        self.bn1 = tf.keras.layers.LayerNormalization(epsilon = 1e-06)
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=(3, 3),
                                        strides=stride,
                                        padding="same")
        self.bn2 = tf.keras.layers.LayerNormalization(epsilon = 1e-06)


    def call(self, inputs, training=None, **kwargs):

        x1 = self.conv1(inputs)
        x1 = self.bn1(x1)
        x2 = self.conv2(inputs)
        x2 = self.bn2(x2)
        output = tf.multiply(x1, tf.sigmoid(x2))

        return output

class pixel_shuffler(tf.keras.layers.Layer):

    def __init__(self, shuffle_size = 2, name = None):
        super(pixel_shuffler, self).__init__()
        self.shuffle_size = shuffle_size

    def call(self, inputs, training=None, **kwargs):
        n = tf.shape(inputs)[0]
        w = tf.shape(inputs)[1]
        c = inputs.get_shape().as_list()[2]
        oc = c // self.shuffle_size
        ow = w * self.shuffle_size

        outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc])
        return outputs

#def pixel_shuffler(inputs, shuffle_size = 2, name = None):
#
#    n = tf.shape(inputs)[0]
#    w = tf.shape(inputs)[1]
#    c = inputs.get_shape().as_list()[2]
#
#    oc = c // shuffle_size
#    ow = w * shuffle_size

#    outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc], name = name)

#    return outputs

class Upsample1d_Block(tf.keras.layers.Layer):
    def __init__(self, filter_num=1024, kernel_size=3, stride=1, shuffle_size = 2):
        super(Upsample1d_Block, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num,
                                            kernel_size=3,
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.LayerNormalization(epsilon = 1e-06)
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_num,
                                            kernel_size=3,
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.LayerNormalization(epsilon = 1e-06)
        self.pixel_shuffler = pixel_shuffler(shuffle_size = 2)
    def call(self, inputs, training=None, **kwargs):
        x1 = self.conv1(inputs)
        x1 = self.pixel_shuffler(x1)
        x1 = self.bn1(x1)

        x2 = self.conv2(inputs)
        x2 = self.pixel_shuffler(x2)
        x2 = self.bn2(x2)
        output = tf.multiply(x1, tf.sigmoid(x2))

        return output


# Construct gated BatchNormalization
def generator_gatecnn(num_features=24, name = None):

    inputs = tf.keras.Input(shape=(num_features, None))
    x = tf.transpose(inputs, perm = [0, 2, 1])
    x1 = tf.keras.layers.Conv1D(filters=128,
                                kernel_size=15,
                                strides=1,
                                padding="same")(x)
    x = tf.keras.layers.Conv1D(filters=128,
                                kernel_size=15,
                                strides=1,
                                padding="same")(x1)
    x = tf.multiply(x1, tf.sigmoid(x))
    x = DownSample_1D(filter_num = 256, kernel_size = 5, stride = 2)(x)
    x = DownSample_1D(filter_num = 512, kernel_size = 5, stride = 2)(x)
    x = BasicBlock_1D(filter_num = 1024, kernel_size = 3, stride = 1)(x)
    x = BasicBlock_1D(filter_num = 1024, kernel_size = 3, stride = 1)(x)
    x = BasicBlock_1D(filter_num = 1024, kernel_size = 3, stride = 1)(x)
    x = BasicBlock_1D(filter_num = 1024, kernel_size = 3, stride = 1)(x)
    x = BasicBlock_1D(filter_num = 1024, kernel_size = 3, stride = 1)(x)
    x = BasicBlock_1D(filter_num = 1024, kernel_size = 3, stride = 1)(x)
    x = Upsample1d_Block(filter_num = 1024, kernel_size = 5, stride = 1, shuffle_size = 2)(x)
    x = Upsample1d_Block(filter_num = 512, kernel_size = 5, stride = 1, shuffle_size = 2)(x)
    x = tf.keras.layers.Conv1D(filters=24,
                                kernel_size=15,
                                strides=1,
                                padding="same")(x)
    x = tf.transpose(x, perm = [0, 2, 1])

    model = tf.keras.models.Model(inputs, x, name=name)
    return model

# Construct Descriminator
def discriminator(num_features=24, name=None):

    inputs = tf.keras.Input(shape=(num_features, 128))
    x = tf.expand_dims(inputs, 0)
    x1 = tf.keras.layers.Conv2D(filters=128,
                                        kernel_size=(3, 3),
                                        strides=[1,2],
                                        padding="same")(x)
    x2 = tf.keras.layers.Conv2D(filters=128,
                                        kernel_size=(3, 3),
                                        strides=[1,2],
                                        padding="same")(x)
    x = tf.multiply(x1, tf.sigmoid(x2))
    x = DownSample_2D(filter_num = 256, kernel_size = [3,3], stride = [2,2])(x)
    x = DownSample_2D(filter_num = 256, kernel_size = [3,3], stride = [2,2])(x)
    x = DownSample_2D(filter_num = 256, kernel_size = [3,3], stride = [1,2])(x)
    x = tf.keras.layers.Dense(units=1, activation = 'sigmoid')(x)

    model = tf.keras.models.Model(inputs, x, name=name)
    return model
