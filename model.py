import tensorflow as tf

N_CLASS = 1000
IMAGE_SHAPE = (224, 224)

def build_siamese_model():
    '''
    Build a siamese model.

    Returns:
        A keras model.
    '''

    def norm_fn(x):
        return tf.keras.backend.l2_normalize(x, axis=1)
    
    return tf.keras.Sequential([
        tf.keras.layers.Dense(512, input_shape=(1280,)),
        tf.keras.layers.Lambda(norm_fn, name='L2Norm')
    ], name='Siamese')

def build_classification_model(top_dropout_rate = 0.2):
    '''
    Build a classification model based on EfficientNetB0. The base model is set
    to non-trainable. A new top layers consisting of a BN, Dropout, 
    and a Dense with softmax activation is appended to the average pooling layer
    of the EfficientNet. 

    Args:
        top_drop_rate: a ```float```.

    Returns:
        A keras model.
    '''
    base_model = tf.keras.applications.EfficientNetB0(
        weights = 'imagenet',
        classes = N_CLASS, 
        input_shape = IMAGE_SHAPE + (3,), 
        pooling = 'avg',
        include_top = False,
    )
    base_model.trainable = False

    x = tf.keras.layers.BatchNormalization()(base_model.output)
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    y = tf.keras.layers.Dense(N_CLASS, activation="softmax", name="pred")(x)
    
    return tf.keras.Model(base_model.input, y, name="EfficientNet")

def fit(model, train_ds, valid_ds, epochs=10, initial_epoch=0, prefix=''):
    '''
    Call model.fit with a logger callback and a checkpoint saving callback.

    Args:
        model: a keras model.
        train_ds: a ```Dataset``` for training.
        valid_ds: a ```Dataset``` for validation.
        epochs: total number of epochs for training.
        initial_epoch: set the starting epoch for the fit. This can affect
                       the optimizer.
        prefix: a ```string``` for naming of output files.

    Returns:
        A history.
    '''
    cp_logger = tf.keras.callbacks.CSVLogger('%s_log.csv'%prefix)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='%s_weights.{epoch:05d}.hdf5'%prefix,
        save_weights_only=True,
        save_best_only=False,
        verbose=1,
    )

    return model.fit(
        train_ds,
        epochs=epochs, 
        verbose=1,
        shuffle=False, 
        callbacks=[cp_callback, cp_logger], 
        initial_epoch=initial_epoch, 
        use_multiprocessing=True,
        validation_data=valid_ds, 
    )
