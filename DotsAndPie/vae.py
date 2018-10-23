from models import *


class VAE:
    def __init__(self, args, dataset, log_path):
        self.args = args
        self.log_path = log_path

        if 'conv' in args.architecture:
            encoder = encoder_conv64
            generator = generator_conv64
        elif 'small' in args.architecture:
            encoder = encoder_conv64small
            generator = generator_conv64small
        elif 'large' in args.architecture:
            encoder = encoder_conv64large
            generator = generator_conv64large
        else:
            encoder = encoder_fc64
            generator = generator_fc64

        # Build the computation graph for training
        self.train_x = tf.placeholder(tf.float32, shape=[None] + dataset.data_dims)
        train_zdist, train_zsample = encoder(self.train_x, args.z_dim)
        # ELBO loss divided by input dimensions
        zkl_per_sample = tf.reduce_sum(-tf.log(train_zdist[1]) + 0.5 * tf.square(train_zdist[1]) +
                                       0.5 * tf.square(train_zdist[0]) - 0.5, axis=1)
        loss_zkl = tf.reduce_mean(zkl_per_sample)
        train_xr = generator(train_zsample)

        # Build the computation graph for generating samples
        self.gen_z = tf.placeholder(tf.float32, shape=[None, args.z_dim])
        self.gen_x = generator(self.gen_z, reuse=True)

        # Negative log likelihood per dimension
        nll_per_sample = tf.reduce_sum(tf.square(self.train_x - train_xr) + 0.5 * tf.abs(self.train_x - train_xr), axis=(1, 2, 3))
        loss_nll = tf.reduce_mean(nll_per_sample)

        self.kl_anneal = tf.placeholder(tf.float32)
        loss_elbo = loss_nll + loss_zkl * args.beta * self.kl_anneal
        self.trainer = tf.train.AdamOptimizer(10 ** args.lr, beta1=0.5, beta2=0.9).minimize(loss_elbo)

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('loss_zkl', loss_zkl),
            tf.summary.scalar('loss_nll', loss_nll),
            tf.summary.scalar('loss_elbo', loss_elbo),
        ])

        self.sample_summary = tf.summary.merge([
            create_display(tf.reshape(self.gen_x, [64]+dataset.data_dims), 'samples'),
            create_display(tf.reshape(train_xr, [64]+dataset.data_dims), 'reconstructions'),
            create_display(tf.reshape(self.train_x, [64]+dataset.data_dims), 'train_samples')
        ])

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.summary_writer = tf.summary.FileWriter(log_path)
        self.sess.run(tf.global_variables_initializer())

        self.idx = 1

    def train_step(self, bx):
        self.sess.run(self.trainer, feed_dict={self.train_x: bx, self.kl_anneal: 1 - math.exp(-self.idx / 50000.0)})

        if self.idx % 100 == 0:
            summary_val = self.sess.run(self.train_summary,
                                        feed_dict={self.train_x: bx, self.kl_anneal: 1 - math.exp(-self.idx / 50000.0)})
            self.summary_writer.add_summary(summary_val, self.idx)

        if self.idx % 2000 == 0:
            summary_val = self.sess.run(self.sample_summary,
                                        feed_dict={self.train_x: bx[:64],
                                                   self.gen_z: sample_z(64, self.args.z_dim, 'gaussian')})
            self.summary_writer.add_summary(summary_val, self.idx)

        self.idx += 1

    def sample(self, batch_size):
        bxg_list = []
        cur_size = 0
        while cur_size < batch_size:
            new_size = min(batch_size - cur_size, 256)
            bxg_list.append(self.sess.run(self.gen_x,
                                          feed_dict={self.gen_z: sample_z(new_size, self.args.z_dim, 'gaussian')}))
            cur_size += new_size
        bxg = np.concatenate(bxg_list, axis=0)
        return bxg

    def save(self):
        saver = tf.train.Saver(
            var_list=[var for var in tf.global_variables() if 'i_net' in var.name or 'g_net' in var.name])
        saver.save(self.sess, os.path.join(self.log_path, "model.ckpt"))