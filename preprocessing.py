import tensorflow as tf


blur = tf.keras.layers.AveragePooling2D(pool_size=(5,5), padding='same')

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
    def prep_y_bl(file_path, image_size=(512, 512)):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.stack([img == 0, img == 100, img == 255], axis=3)
        img = tf.reshape(img, shape=(img.shape[0], img.shape[1], 3))
        img = tf.cast(img, dtype=tf.float32)    
        img = tf.image.resize(img, size=image_size, method='bilinear')
        return img

    @staticmethod
    @tf.function
    def prep_y_s(file_path, image_size=(512, 512), smoothing=2):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.stack([img == 0, img == 100, img == 255], axis=3)
        img = tf.reshape(img, shape=(img.shape[0], img.shape[1], 3))
        img = tf.cast(img, dtype=tf.float32)    
        img = tf.image.resize(img, size=(int(image_size[0]/smoothing), int(image_size[1]/smoothing)) , method='bilinear')
        img = tf.image.resize(img, size=image_size, method='bilinear')
        return img

    @staticmethod
    @tf.function
    def blur_labels(y):
        img = tf.reshape(y, shape=(1, y.shape[0], y.shape[1], y.shape[2]))
        img = blur(img)
        img = tf.squeeze(img)
        img = tf.image.resize(img, size=(y.shape[0], y.shape[1]), method='bilinear')
        return img
        

    @staticmethod
    @tf.function
    def smooth_labels(y, factor=0.1):
        ys = y*(1 - factor)
        ys += (factor / y.shape[-1])
        return ys

    @staticmethod
    def get_tf_dataset(ds, image_size, batch_size, shard=True):
        def prep_ds(x, y):
                px = Prep.prep_x(x, image_size=image_size)
                py = Prep.prep_y(y, image_size=image_size)
                return px, py

        tf_ds = tf.data.Dataset.from_tensor_slices((ds.raw_path, ds.mask_path))

        options = tf.data.Options()
        
        if shard:
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        else:
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        tf_ds = tf_ds.with_options(options)

        tf_ds = tf_ds.map(
            prep_ds,
            # num_parallel_calls=tf.data.AUTOTUNE,
        )
        tf_ds = tf_ds.batch(batch_size, drop_remainder=True)
        tf_ds = tf_ds.prefetch(tf.data.AUTOTUNE)

        return tf_ds
