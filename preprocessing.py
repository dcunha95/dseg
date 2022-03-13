import tensorflow as tf


class Prep:
    @staticmethod
    @tf.function
    def prep_x(file_path, image_size=(512, 512)):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, size=image_size)
        return img

    @staticmethod
    @tf.function
    def prep_y(file_path, image_size=(512, 512)):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, size=image_size, method='nearest')
        img = tf.stack([img == 0, img == 100, img == 255], axis=3)
        img = tf.reshape(img, shape=(image_size[0], image_size[1], 3))
        img = tf.cast(img, dtype=tf.float32)
        return img

    @staticmethod
    @tf.function
    def smooth_labels(y, factor=0.1):
        ys = y*(1 - factor)
        ys += (factor / y.shape[-1])
        return ys
