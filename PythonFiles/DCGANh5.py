import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import glob
import matplotlib.pyplot as plt


class DCGAN():
    def __init__(self):
        self.image_path = glob.glob('E:\\tensorflow_learning\\animeFaces\\data\\*.png')
        # self.checkpoint_path = r'E:\PyCharm\Projects\DCGANh5'
        self.image_ds = tf.data.Dataset.from_tensor_slices(self.image_path)  # 建立一个dataset
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE  # 设置为tf.data.experimental.AUTOTUNE,它将提示tf.data runtime在运行时动态地调整值
        self.image_ds = self.image_ds.map(self.load_preprosess_image, num_parallel_calls=self.AUTOTUNE)
        self.BTATH_SIZE = 64
        self.image_count = len(self.image_path)
        self.image_ds = self.image_ds.shuffle(self.image_count).batch(self.BTATH_SIZE)  # 乱序
        self.image_ds = self.image_ds.prefetch(self.AUTOTUNE)  # 提前从数据集中取出若干数据放到内存中，这样可以使在gpu计算时，cpu通过处理数据，从而提高训练的速度。
        self.noise = tf.random.normal([1, 100])
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # 计算交叉熵
        self.generator_optimizer = keras.optimizers.Adam(1e-5)
        self.discriminator_optimizer = keras.optimizers.Adam(1e-5)
        self.EPOCHS = 2000
        self.noise_dim = 100
        self.num_exp_to_generate = 4
        self.seed = tf.random.normal([self.num_exp_to_generate, self.noise_dim])
        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()
        # self.checkpoint_prefix = os.path.join(self.checkpoint_path, "ckpt")
        # self.checkpoint = tf.train.Checkpoint(
        #   generator_optimizer=self.generator_optimizer,
        #  discriminator_optimizer=self.discriminator_optimizer,
        # generator=self.generator,
        # discriminator=self.discriminator)

    # 生成器(反卷积)
    def generator_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(8 * 8 * 256,
                               input_shape=(100,),
                               use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Reshape((8, 8, 256)))  # 8*8*256

        model.add(layers.Conv2DTranspose(128, (5, 5),
                                         strides=(1, 1),
                                         padding='same',
                                         use_bias=False))
        model.add(layers.BatchNormalization())  # 8*8*128
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5),
                                         strides=(2, 2),
                                         padding='same',
                                         use_bias=False))
        model.add(layers.BatchNormalization())  # 16*16*128
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(32, (5, 5),
                                         strides=(2, 2),
                                         padding='same',
                                         use_bias=False))
        model.add(layers.BatchNormalization())  # 32*32*32
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5),
                                         strides=(2, 2),
                                         padding='same',
                                         use_bias=False,
                                         activation='tanh'))  # 64*64*3

        return model

    # 判别器(卷积)
    def discriminator_model(self):
        model = keras.Sequential()
        model.add(layers.Conv2D(32, (5, 5),
                                strides=(2, 2),
                                padding='same',
                                input_shape=(64, 64, 3)))  # 32*32*32
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))  # 希望判别器判别不是那么精确

        model.add(layers.Conv2D(64, (5, 5),
                                strides=(2, 2),
                                padding='same'))  # 16*16*64
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5),
                                strides=(2, 2),
                                padding='same'))  # 8*8*128
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, (5, 5),
                                strides=(2, 2),
                                padding='same'))  # 4*4*256
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.GlobalAveragePooling2D())

        model.add(layers.Dense(1024))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(1))

        return model

    # 编写损失函数，定义优化器
    def discriminator_loss(self, real_out, fake_out):
        real_loss = self.cross_entropy(tf.ones_like(real_out), real_out)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_out), fake_out)
        return real_loss + fake_loss

    def generator_loss(self, fake_out):
        return self.cross_entropy(tf.ones_like(fake_out), fake_out)

    # 加载与预处理
    def load_preprosess_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)  # 对图片进行解码，channels=3，为彩色图片
        # image=tf.image.resize_with_crop_or_pad(image,256,256)   规范大小
        image = tf.cast(image, tf.float32)  # 规范数据类型
        image = (image / 127.5) - 1  # 归一化
        return image

    # 定义批次训练函数
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BTATH_SIZE, self.noise_dim])

        with tf.GradientTape()as gen_tape, tf.GradientTape()as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    # 可视化
    def generate_and_save_images(self, model, epoch, test_input):
        predictions = model(test_input, training=False)
        fig = plt.figure(figsize=(6, 6))
        for i in range(predictions.shape[0]):
            plt.subplot(2, 2, i + 1)  # 从1开始
            plt.imshow((predictions[i, :, :, :] + 1) / 2)
            plt.axis('off')
        if epoch % 20 == 0:
            plt.savefig(f'E:\PyCharm\Projects\DCGANh5\{epoch}.png')
        plt.show()

    def train(self):
        for epoch in range(self.EPOCHS):
            for image_batch in self.image_ds:
                self.train_step(image_batch)
                # print('.',end='')
            # if epoch % 10 == 0:
            # print(epoch)
            self.generate_and_save_images(self.generator, epoch + 1, self.seed)
            self.generator.save('./save/DCGAN_cartoon_64_{:04d}.h5'.format(epoch))
            # self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        self.generate_and_save_images(self.generator, epoch, self.seed)


# if __name__ == '__main__':
#     gan = DCGAN()
#     gan.train()
