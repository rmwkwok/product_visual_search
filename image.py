import tensorflow as tf

IMAGE_SHAPE = (224, 224)

def augmentation(images, labels):
    '''
    Return augmented images and corresponding labels. The order of images
    is determined by ORDER_BY_AUG.

    Args:
        images: An 4-D `float` Tensor of shape
                (batch_size, )+IMAGE_SHAPE+(3, ).
        labels: An 1-D `int` Tensor of shape (batch_size, ).

    Returns: 
        return_images: An 4-D `float` Tensor of shape
                       (batch_size*num_augment, )+IMAGE_SHAPE+(3, ).
        return_labels: An 1-D `int` Tensor of shape (batch_size*num_augment).
    '''

    # ORDER_BY_AUG
    # True: augmented images are ordered by augmentations, i.e. 
    # image1_aug1, image2_aug1, image3_aug1, ..., image1_aug2, image2_aug2, ...
    #
    # False: augmented images are orders by images, i.e.
    # image1_aug1, image1_aug2, image3_aug3, ..., imageN_aug1, imageN_aug2, ...
    ORDER_BY_AUG = False

    augmented = [
              images,
              tf.image.flip_up_down(images),
              tf.image.flip_left_right(images),
              tf.image.rot90(images, k=1),
              tf.image.rot90(images, k=2),
              tf.image.rot90(images, k=3),
    ]

    if ORDER_BY_AUG:
        return_labels = tf.tile(labels, [len(augmented)])
        return_images = tf.concat(augmented, axis=0)
    else:
        return_labels = tf.repeat(labels, [len(augmented)])
        return_images = tf.reshape(
            tf.transpose(augmented, perm=(1,0,2,3,4)), 
            (-1,)+IMAGE_SHAPE+(3,)
        )

    return return_images, return_labels

def get_perimeter_mean(channel):
    '''
    Get the mean value of perimeter's pixels. Thickness of the perimeter is 
    defined with P_THICK in terms of pixel value.

    Args:
        channel: An 2-D `float` Tensor of shape (h, w).

    Returns: 
        An `int` as the mean pixel value
    '''
    P_THICK = 1
    return tf.cast(tf.reduce_mean(tf.concat([
        tf.reshape(channel[: P_THICK, :], [-1]),
        tf.reshape(channel[-P_THICK:, :], [-1]),
        tf.reshape(channel[:, : P_THICK], [-1]),
        tf.reshape(channel[:, -P_THICK:], [-1]),
    ], axis=0)), channel.dtype)

def pad_and_resize(image, resize_shape=IMAGE_SHAPE):
    '''
    Pad to make the shorter side to have the same length as the longer side.

    Args:
        image: An 3-D `int` Tensor of shape (h, w, 3).
        resize_shape: An 1-D `int` Tensor of (h, w, 3).

    Returns: 
        An 3-D `int` Tensor
    '''
    h, w, d = tf.unstack(tf.shape(image))
    target_w = target_h = tf.maximum(h, w)

    pad_up = (target_h - h)//2
    pad_dn = target_h - h - pad_up
    pad_lf = (target_w - w)//2
    pad_rt = target_w - w - pad_lf
    paddings = tf.reshape([pad_up, pad_dn, pad_lf, pad_rt, 0, 0], [3,2])

    c0, c1, c2 = image[:,:,0:1], image[:,:,1:2], image[:,:,2:3]
    image = tf.concat([
        tf.pad(c0, paddings, constant_values=get_perimeter_mean(c0)),
        tf.pad(c1, paddings, constant_values=get_perimeter_mean(c1)),
        tf.pad(c2, paddings, constant_values=get_perimeter_mean(c2)),
    ], axis=2)
    return tf.image.resize(image, resize_shape)