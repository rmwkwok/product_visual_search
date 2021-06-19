import tensorflow as tf

def _feature_int(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))

def _feature_byte(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _feature_float_array(val):
    return tf.train.Feature(float_list=tf.train.FloatList(value=val))

def _serialize_feature(feature):
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)
    return example.SerializeToString()

def tfr_parser_emb_idx(tfrecord):
    example = tf.io.parse_single_example(
        tfrecord,
        {'emb': tf.io.FixedLenFeature([1280], tf.float32),
         'idx': tf.io.FixedLenFeature([], tf.int64)}
    )
    return example['emb'], tf.cast(example['idx'], tf.int32)

def tfr_parser_emb(tfrecord):
    example = tf.io.parse_single_example(
        tfrecord,
        {'emb': tf.io.FixedLenFeature([1280], tf.float32)}
    )
    return example['emb']

def tfr_parser_image_label(tfrecord):
    '''
    Retrieve the image and the integer label from a TFRecord. 
    The image is resized to IMAGE_SHAPE.

    Args:
        tfrecord.

    Returns:
        image: An 3-D `float` Tensor.
        label: A `int` scalar.
    '''
    example = tf.io.parse_single_example(
        tfrecord, 
        {'image': tf.io.FixedLenFeature([], tf.string),
         'label': tf.io.FixedLenFeature([], tf.int64)},
    )

    image = tf.image.decode_jpeg(example['image'], channels=3)[...,::-1]
    image = tf.image.resize(image, IMAGE_SHAPE)
    label = tf.cast(example['label'], tf.int32)
    return image, label

def tfr_parser_image(tfrecord):
    '''
    Retrieve the image from a TFRecord. The image is resized to IMAGE_SHAPE.

    Args:
        tfrecord.

    Returns:
        image: An 3-D `float` Tensor.
    '''
    example = tf.io.parse_single_example(
        tfrecord, {'image': tf.io.FixedLenFeature([], tf.string)}
    )

    image = tf.image.decode_jpeg(example['image'], channels=3)[...,::-1]
    image = tf.image.resize(image, IMAGE_SHAPE)
    return image