from utils import *
Bernoulli = tf.contrib.distributions.Bernoulli


def sample_z(batch_size, z_dim, noise='gaussian'):
    if 'gaussian' in noise:
        return np.random.normal(0, 1, [batch_size, z_dim])
    elif 'bernoulli' in noise:
        return (np.random.normal(0, 1, [batch_size, z_dim]) > 0).astype(np.float)
    return None


# Encoders


def encoder_conv28(x, z_dim):
    with tf.variable_scope('i_net'):
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        mean = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        sample = mean + tf.multiply(stddev, tf.random_normal(tf.stack([tf.shape(x)[0], z_dim])))
        return [mean, stddev], sample


def encoder_conv64(x, z_dim):
    with tf.variable_scope('i_net'):
        conv = conv2d_bn_lrelu(x, 64, 4, 2)
        conv = conv2d_bn_lrelu(conv, 128, 4, 2)
        conv = conv2d_bn_lrelu(conv, 192, 4, 2)
        conv = conv2d_bn_lrelu(conv, 256, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 1024)
        mean = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        sample = mean + tf.multiply(stddev, tf.random_normal(tf.stack([tf.shape(x)[0], z_dim])))
        return [mean, stddev], sample


def encoder_conv64small(x, z_dim):
    with tf.variable_scope('i_net'):
        conv = conv2d_bn_lrelu(x, 32, 4, 2)
        conv = conv2d_bn_lrelu(conv, 64, 4, 2)
        conv = conv2d_bn_lrelu(conv, 96, 4, 2)
        conv = conv2d_bn_lrelu(conv, 128, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 512)
        mean = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        sample = mean + tf.multiply(stddev, tf.random_normal(tf.stack([tf.shape(x)[0], z_dim])))
        return [mean, stddev], sample


def encoder_conv64large(x, z_dim):
    with tf.variable_scope('i_net'):
        conv = conv2d_bn_lrelu(x, 128, 4, 2)
        conv = conv2d_bn_lrelu(conv, 256, 4, 2)
        conv = conv2d_bn_lrelu(conv, 384, 4, 2)
        conv = conv2d_bn_lrelu(conv, 512, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc,  2048)
        mean = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        sample = mean + tf.multiply(stddev, tf.random_normal(tf.stack([tf.shape(x)[0], z_dim])))
        return [mean, stddev], sample



def encoder_fc64(x, z_dim):
    with tf.variable_scope('i_net'):
        fc = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
        fc = fc_bn_lrelu(fc, 1024)
        fc = fc_bn_lrelu(fc, 1024)
        fc = fc_bn_lrelu(fc, 1024)
        fc = fc_bn_lrelu(fc, 1024)
        mean = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        sample = mean + tf.multiply(stddev, tf.random_normal(tf.stack([tf.shape(x)[0], z_dim])))
        return [mean, stddev], sample


# Generators


def generator_conv64(z, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_bn_relu(z, 1024)
        fc = fc_bn_relu(fc, 4*4*256)
        conv = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 4, 4, 256]))
        conv = conv2d_t_bn_relu(conv, 192, 4, 2)
        conv = conv2d_t_relu(conv, 128, 4, 2)
        conv = conv2d_t_relu(conv, 128, 4, 1)
        conv = conv2d_t_relu(conv, 64, 4, 2)
        output = tf.contrib.layers.convolution2d_transpose(conv, 3, 4, 2, activation_fn=tf.sigmoid)
        return output


def generator_conv64small(z, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_relu(z, 768)
        fc = fc_relu(fc, 4*4*192)
        conv = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 4, 4, 192]))
        conv = conv2d_t_relu(conv, 128, 4, 2)
        conv = conv2d_t_relu(conv, 96, 4, 2)
        conv = conv2d_t_relu(conv, 96, 4, 1)
        conv = conv2d_t_relu(conv, 48, 4, 2)
        output = tf.contrib.layers.convolution2d_transpose(conv, 3, 4, 2, activation_fn=tf.sigmoid)
        return output


def generator_conv64large(z, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_relu(z, 2048)
        fc = fc_relu(fc, 4*4*512)
        conv = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 4, 4, 512]))
        conv = conv2d_t_relu(conv, 384, 4, 2)
        conv = conv2d_t_relu(conv, 256, 4, 2)
        conv = conv2d_t_relu(conv, 256, 4, 1)
        conv = conv2d_t_relu(conv, 128, 4, 2)
        output = tf.contrib.layers.convolution2d_transpose(conv, 3, 4, 2, activation_fn=tf.sigmoid)
        return output


def generator_fc64(z, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_bn_relu(z, 1024)
        fc = fc_bn_relu(fc, 1024)
        fc = fc_bn_relu(fc, 1024)
        fc = fc_bn_relu(fc, 1024)
        fc = tf.contrib.layers.fully_connected(fc, 64*64*3, activation_fn=tf.sigmoid)
        output = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 64, 64, 3]))
        return output


def generator_conv28(z, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_relu(z, 1024)
        fc = fc_relu(fc, 7*7*128)
        fc = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 7, 7, 128]))
        conv = conv2d_t_relu(fc, 64, 4, 2)
        conv = conv2d_t_relu(conv, 64, 4, 1)
        output = tf.contrib.layers.convolution2d_transpose(conv, 1, 4, 2, activation_fn=tf.sigmoid)
        return output


# Discriminators


def discriminator_conv28(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        fc2 = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
        return fc2


def discriminator_conv64(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv = conv2d_lrelu(x, 64, 4, 2)
        conv = conv2d_lrelu(conv, 128, 4, 2)
        conv = conv2d_lrelu(conv, 192, 4, 2)
        conv = conv2d_lrelu(conv, 256, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 1024)
        fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=tf.identity)
        return fc


def discriminator_conv64large(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv = conv2d_lrelu(x, 128, 4, 2)
        conv = conv2d_lrelu(conv, 256, 4, 2)
        conv = conv2d_lrelu(conv, 384, 4, 2)
        conv = conv2d_lrelu(conv, 512, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 2048)
        fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=tf.identity)
        return fc


def discriminator_conv64small(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv = conv2d_lrelu(x, 48, 4, 2)
        conv = conv2d_lrelu(conv, 96, 4, 2)
        conv = conv2d_lrelu(conv, 128, 4, 2)
        conv = conv2d_lrelu(conv, 192, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 768)
        fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=tf.identity)
        return fc


def discriminator_fc64(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 1024)
        fc = fc_lrelu(fc, 1024)
        fc = fc_lrelu(fc, 1024)
        fc = fc_lrelu(fc, 1024)
        fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=tf.identity)
        return fc
