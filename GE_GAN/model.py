#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 18:43
# @Author  : Chenchen Wei
# @Description:
from GE_GAN.get_data import *
from GE_GAN.layers import *


class GE_GAN(object):
    def _call(self):
        self.scaler, self.train_x, self.train_y, self.test_x, self.test_y = self.all_data
        self.batch_nums = int(self.train_x.shape[0] / self.batch_size + 1)
        self.tr_bat_nums = self.batch_size - self.train_x.shape[0] % self.batch_size
        self.input_dim = self.train_x.shape[-1]
        self.output_dim = self.train_y.shape[-1]

    def __init__(self, sess, all_data,
                 ge_hidden_lists,
                 dis_hidden_lists,
                 reconstruction_coefficient=100,
                 target_nums=1,
                 flags='x',  # if want to add noise_z, change flags='x_z'
                 epoch=1,
                 batch_size=2,
                 drop_rate=0,
                 learning_rate=5e-5,
                 noise_dim=50,
                 save_model=False,  # if save_model is True, should give the save_model_path
                 save_model_path=None):
        self.sess = sess
        self.all_data = all_data
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.learning_rate = learning_rate
        self.save_model = save_model
        self.save_model_path = save_model_path
        self.flags = flags
        self.target_nums = target_nums  # the number of target_segments
        self.epoch = epoch
        self.ge_hidden_lists = ge_hidden_lists
        self.dis_hidden_lists = dis_hidden_lists
        self.noise_dim = noise_dim
        self.reconstruction_coefficient = reconstruction_coefficient

    def generate(self, inputs, noise_z, reuse=False):
        """Generator, to generate the data of the target_segments"""
        with tf.variable_scope('generate', reuse=reuse):
            if self.flags == 'x_z':  # add the noise in the input of the GE-GAN
                inputs = tf.concat([inputs, noise_z], axis=0)
                h1 = Linear('ge_h1',
                            input_dim=inputs.shape[-1] + noise_z.shape[-1],
                            output_dim=self.ge_hidden_lists[0])(inputs)
            else:
                h1 = Linear('ge_h1',
                            input_dim=inputs.shape[-1],
                            output_dim=self.ge_hidden_lists[0])(inputs)
            h2 = Linear('ge_h2',
                        input_dim=self.ge_hidden_lists[0],
                        output_dim=self.ge_hidden_lists[1])(h1)
            h3 = Linear('ge_h3',
                        input_dim=self.ge_hidden_lists[1],
                        output_dim=self.ge_hidden_lists[2], )(h2)
            out = Linear('ge_out',
                         input_dim=self.ge_hidden_lists[2],
                         output_dim=self.train_y.shape[-1],
                         act=tf.nn.sigmoid)(h3)
            return out

    def discriminator(self, inputs, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            h1 = Linear('dis_h1',
                        input_dim=inputs.shape[-1],
                        output_dim=self.dis_hidden_lists[0])(inputs)
            h2 = Linear('dis_h2',
                        input_dim=self.dis_hidden_lists[0],
                        output_dim=self.dis_hidden_lists[1])(h1)
            h3 = Linear('dis_h3',
                        input_dim=self.dis_hidden_lists[1],
                        output_dim=self.dis_hidden_lists[2])(h2)
            out_logit = Linear('dis_logit',
                               input_dim=self.dis_hidden_lists[2],
                               output_dim=1,
                               act=lambda x: x)(h3)  # not use activation function,
            out_prob = tf.nn.sigmoid(out_logit)
            return out_prob, out_logit

    def get_batch(self, i, batch_nums, data):
        """ Return a batch of data"""
        if i != batch_nums - 1:
            return data[i * self.batch_size:(i + 1) * self.batch_size, :]
        else:
            nums = self.tr_bat_nums
            temp = data[i * self.batch_size:, :]
            tm_zeros = np.zeros(shape=(nums, data.shape[-1]))
            concats = np.concatenate((temp, tm_zeros), axis=0)
            return concats

    def _build(self):
        self._call()
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim])  # generate data
        self.x_real = tf.placeholder(tf.float32, shape=[None, self.output_dim])  # observed data
        self.noise_z = tf.placeholder(tf.float32, shape=[None, self.noise_dim])  # noise

        D_real, D_logit_real = self.discriminator(self.x_real, reuse=False)
        G_samples = self.generate(self.x, self.noise_z, reuse=False)  # generate data by generator
        D_fake, D_logit_fake = self.discriminator(G_samples, reuse=True)
        d_loss_real = -tf.reduce_mean(D_logit_real)
        d_loss_fake = tf.reduce_mean(D_logit_fake)

        mse_loss = tf.reduce_mean(tf.square(G_samples - self.x_real))  # reconstruction loss
        self.d_loss = d_loss_real + d_loss_fake
        self.g_loss = - d_loss_fake + self.reconstruction_coefficient * mse_loss

        self.t_vars = tf.trainable_variables()  # trainable_variables
        d_vars = [var for var in self.t_vars if 'dis' in var.name]
        g_vars = [var for var in self.t_vars if 'ge' in var.name]

        self.d_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]  # clip D values
        self.d_optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate
                                                 ).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate
                                                 ).minimize(self.g_loss, var_list=g_vars)

        self.generate_data = self.generate(self.x, self.noise_z, reuse=True)

    def train(self):
        self._build()
        tf.global_variables_initializer().run()
        start_time = time.time()
        self.d_loss_sum, self.g_loss_sum = [], []  # Record loss values
        saver = tf.train.Saver(max_to_keep=1)
        for epoch in range(self.epoch):
            d_bat, g_bat = [], []
            for i in range(self.batch_nums):
                x_train = self.get_batch(i, self.batch_nums, self.train_x)
                y_train = self.get_batch(i, self.batch_nums, self.train_y)
                zz = sample_Z(self.batch_size, self.noise_dim)
                d_per = []
                for _ in range(5):  # repeat 5 times for discriminator
                    _, D_loss, _ = self.sess.run([self.d_optim, self.d_loss, self.d_clip],
                                                 feed_dict={self.x_real: y_train,
                                                            self.x: x_train,
                                                            self.noise_z: zz})
                    d_per.append(D_loss)

                _, G_loss = self.sess.run([self.g_optim, self.g_loss],
                                          feed_dict={self.x_real: y_train,
                                                     self.x: x_train,
                                                     self.noise_z: zz})
                d_bat.append(np.mean(np.asarray(d_per)))
                g_bat.append(G_loss)
            d_epoch, g_epoch = np.mean(np.asarray(d_bat)), np.mean(np.asarray(g_bat))
            self.d_loss_sum.append(d_epoch)
            self.g_loss_sum.append(g_epoch)
            if epoch % 20 == 0:
                use_time = (time.time() - start_time) / 60
                print('Epoch:[{}], G_Loss:{:.4f}, D_Loss:{:.8f}, usu_time:{:.2f}min'
                      .format(epoch, g_epoch, d_epoch, use_time))

        if self.save_model:
            if not os.path.exists(self.save_model_path):
                os.makedirs(self.save_model_path)
            saver.save(self.sess, self.save_model_path + '/GE_GAN')  # Save model
        test_z = sample_Z(self.test_x.shape[0], self.noise_dim)
        gene = self.sess.run(self.generate_data,
                             feed_dict={self.x: self.test_x,
                                        self.noise_z: test_z})
        gene = gene[:, -self.target_nums:] #the generated data of the target_segments
        true = self.test_y[:, -self.target_nums:]
        gene = self.scaler.inverse_transform(gene)
        true = self.scaler.inverse_transform(true)

        return gene, true
