import cifarloader
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Union
tf.config.set_visible_devices([], device_type='GPU')



MEAN_RGB = jnp.array([0.4914 * 255, 0.4822 * 255, 0.4465 * 255]).reshape(1, 1, 3)
STDDEV_RGB = jnp.array([0.2470 * 255, 0.2435 * 255, 0.2616 * 255]).reshape(1, 1, 3)

@partial(jax.jit, static_argnums=(0))
def process_cifar_batch(dshape : tuple, x, y):
    x = x.reshape((len(x), ) + dshape)
    x = (x - MEAN_RGB) / STDDEV_RGB
    yhot = jax.nn.one_hot(y, 10)
    return x, yhot

@jax.jit
def bernoulli_horizontal_flip(x : jnp.ndarray, key : jax.random.PRNGKey):
    should_flip = jax.random.randint(key=key, minval=0, maxval=1, shape=(1, ))
    x_horizontal_flipped = jnp.flip(x, axis=1)
    return should_flip * x_horizontal_flipped + (1 - should_flip) * x

@partial(jax.jit, static_argnums=1)
def pad_image(x : jnp.ndarray, pad_size : int = 4):
    return jnp.pad(x, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')

@partial(jax.jit, static_argnums=2)
def random_symmetric_crop(x : jnp.ndarray, key : jax.random.PRNGKey, crop_size : int = 32):
    height, width, nchannels = x.shape
    hkey, wkey = jax.random.split(key, 2)

    max_height = height - crop_size
    max_width = width - crop_size

    height_break = jax.random.randint(hkey, shape=(), minval=0, maxval=max_height + 1)
    width_break = jax.random.randint(wkey, shape=(), minval=0, maxval=max_width + 1)

    start_indices = (height_break, width_break, 0)
    slice_sizes = (crop_size, crop_size, nchannels)

    return jax.lax.dynamic_slice(x, start_indices, slice_sizes)

@jax.jit
def augment_processed_cifar_image(x : jnp.ndarray, key : jax.random.PRNGKey):
    key1, key2 = jax.random.split(key)
    x = pad_image(x, pad_size=4)
    x = bernoulli_horizontal_flip(x, key1)
    x = random_symmetric_crop(x, key2, crop_size=32)
    return x

augment_processed_cifar_batch = jax.vmap(augment_processed_cifar_image)


@partial(jax.jit, static_argnums=(0))
def process_cifar_masked_batch(dshape : tuple, x, y, batch_ids, ds_mask):
    x, yhot = process_cifar_batch(dshape, x, y)
    batch_mask = ds_mask[batch_ids]
    return batch_mask, (x, yhot)


@jax.jit
def fuse(Xl, Yl, idsl, Xu, Yu, idsu):
    X = jnp.concatenate((Xl, Xu), axis=0)
    Y = jnp.concatenate((Yl, Yu), axis=0)
    ids = jnp.concatenate((idsl, idsu), axis=0)
    return ids, (X, Y)


@dataclass
class CIFAR10ClustSupervised:
    bs : int = 64
    testval_bs : int = 64
    reshuffle : bool = True
    backbone : str = 'ResNet50'
    only_downstream : bool = False

    def __post_init__(self):
        super().__init__()

        if self.backbone is not None:
            self.dshape = (32, 32, 3)
        else:
            raise ValueError

        loader = cifarloader.FS_CIFAR10_Loader()

        if not self.only_downstream:
            self.ds_train = loader.get_train_dataset(bs=self.bs, reshuffle=self.reshuffle)
            self.ds_val = loader.get_valset(bs=-1)
            self.ds_test = loader.get_testset(bs=-1)

            # start an iterator for train set
            self.train_iterator = iter(tfds.as_numpy(self.ds_train))
            self.val_iterator = iter(tfds.as_numpy(self.ds_val))
            self.test_iterator = iter(tfds.as_numpy(self.ds_test))

            # process the validation and test set (only required once)
            X_val, Yhot_val= self.batch_process_valtest(loader.get_valset(bs=self.testval_bs))
            self.batched_valset = (X_val, Yhot_val)

            X_test, Yhot_test = self.batch_process_valtest(loader.get_testset(bs=self.testval_bs))
            self.batched_testset = (X_test, Yhot_test)

        self.ds_downstream = loader.get_downstream(bs=-1, reshuffle=self.reshuffle)
        # start an iterator for downstream
        self.downstream_iterator = iter(tfds.as_numpy(self.ds_downstream))


    def next_train(self, augment=False, key=None):
        try:
            (x, y) = next(self.train_iterator)
            if x.shape[0] != self.bs:
                self.train_iterator = iter(tfds.as_numpy(self.ds_train))
                (x, y) = next(self.train_iterator)

        except StopIteration:
            self.train_iterator = iter(tfds.as_numpy(self.ds_train))
            (x, y) = next(self.train_iterator)

        x, yhot = process_cifar_batch(self.dshape, x, y)
        if augment:
            keys = jax.random.split(key, x.shape[0])
            x = augment_processed_cifar_batch(x, keys)
        return x, yhot

    def next_val(self):
        try:
            (x, y) = next(self.val_iterator)
        except StopIteration:
            self.val_iterator = iter(tfds.as_numpy(self.ds_val))
            (x, y) = next(self.val_iterator)
        return process_cifar_batch(self.dshape, x, y)


    def next_test(self):
        try:
            (x, y) = next(self.test_iterator)
        except StopIteration:
            self.test_iterator = iter(tfds.as_numpy(self.ds_test))
            (x, y) = next(self.test_iterator)
        return process_cifar_batch(self.dshape, x, y)


    def next_downstream(self):
        try:
            (x, y) = next(self.downstream_iterator)
        except StopIteration:
            self.downstream_iterator = iter(tfds.as_numpy(self.ds_downstream))
            (x, y) = next(self.downstream_iterator)
        return process_cifar_batch(self.dshape, x, y)

    def get_batched_val(self):
        return self.batched_valset

    def get_batched_test(self):
        return self.batched_testset

    def batch_process_valtest(self, ds):
        XYs = [(X, Y) for (X, Y) in iter(tfds.as_numpy(ds))]
        Xs = [X[np.newaxis, ...] for (X, _) in XYs]
        Ys = [Y[np.newaxis, ...] for (_, Y) in XYs]

        # drop last batch if the size is irregular
        if Xs[-1].shape[1] != Xs[-2].shape[1]:
            Xs = Xs[:-1]
            Ys = Ys[:-1]
        Xs = np.concatenate(Xs, axis=0)
        Ys = np.concatenate(Ys, axis=0)
        return jax.vmap(process_cifar_batch, in_axes=(None, 0, 0))(self.dshape, Xs, Ys)





@dataclass
class CIFAR10ClustSemiSupervised:

    bs : int = 64
    testval_bs : int = 64
    reshuffle : bool = True
    backbone : str = 'CNN'
    only_downstream : bool = False
    N_labeled : int = 100
    K : int = 0
    lbs : int = 32
    ubs : Union[int, None]  = None

    def __post_init__(self):
        super().__init__()

        assert self.lbs <= self.bs
        if self.ubs is None:
            self.ubs = self.bs - self.lbs


        if self.backbone is not None:
            self.dshape = (32, 32, 3)
        else:
            raise ValueError

        loader = cifarloader.SS_CIFAR10_Loader()

        if not self.only_downstream:
            self.lds, self.uds, self.mask = loader.get_train_datasets(K=self.K,
                                             lbs=self.lbs,
                                             ubs=self.ubs,
                                             N_labeled=self.N_labeled,
                                             reshuffle=self.reshuffle)

            self.ds_val = loader.get_valset(bs=-1)
            self.ds_test = loader.get_testset(bs=-1)

            # start an iterator the data sets 
            self.lds_iterator = iter(tfds.as_numpy(self.lds))
            self.uds_iterator = iter(tfds.as_numpy(self.uds))
            self.val_iterator = iter(tfds.as_numpy(self.ds_val))
            self.test_iterator = iter(tfds.as_numpy(self.ds_test))

            # process the validation and test set (only required once)
            X_val, Yhot_val= self.batch_process_valtest(loader.get_valset(bs=self.testval_bs))
            self.batched_valset = (X_val, Yhot_val)

            X_test, Yhot_test = self.batch_process_valtest(loader.get_testset(bs=self.testval_bs))
            self.batched_testset = (X_test, Yhot_test)


        self.ds_downstream = loader.get_downstream(bs=-1, reshuffle=self.reshuffle)
        # start an iterator for downstream
        self.downstream_iterator = iter(tfds.as_numpy(self.ds_downstream))


    def next_train(self, augment=False, key=None):
        try:
            idsl, (xl, yl) = next(self.lds_iterator)
            if xl.shape[0] != self.lbs:
                self.lds_iterator = iter(tfds.as_numpy(self.lds))
                idsl, (xl, yl) = next(self.lds_iterator)
        except StopIteration:
            self.lds_iterator = iter(tfds.as_numpy(self.lds))
            idsl, (xl, yl) = next(self.lds_iterator)

        try:
            idsu, (xu, yu) = next(self.uds_iterator)
            if xu.shape[0] != self.ubs:
                self.uds_iterator = iter(tfds.as_numpy(self.uds))
                idsu, (xu, yu) = next(self.uds_iterator)
        except StopIteration:
            self.uds_iterator = iter(tfds.as_numpy(self.uds))
            idsu, (xu, yu) = next(self.uds_iterator)

        ids, (x, y) = fuse(xl, yl, idsl, xu, yu, idsu)

        batch_mask, (x, yhot) = process_cifar_masked_batch(self.dshape, x, y, ids, self.mask)

        if augment:
            keys = jax.random.split(key, x.shape[0])
            x = augment_processed_cifar_batch(x, keys)
        return batch_mask, (x, yhot)


    def next_labeled(self, augment=False, key=None):
        try:
            idsl, (xl, yl) = next(self.lds_iterator)
        except StopIteration:
            self.lds_iterator = iter(tfds.as_numpy(self.lds))
            idsl, (xl, yl) = next(self.lds_iterator)

        x, yhot = process_cifar_batch(self.dshape, xl, yl)
        if augment:
            keys = jax.random.split(key, x.shape[0])
            x = augment_processed_cifar_batch(x, keys)
        return x, yhot




    def next_val(self):
        try:
            (x, y) = next(self.val_iterator)
        except StopIteration:
            self.val_iterator = iter(tfds.as_numpy(self.ds_val))
            (x, y) = next(self.val_iterator)
        return process_cifar_batch(self.dshape, x, y)


    def next_test(self):
        try:
            (x, y) = next(self.test_iterator)
        except StopIteration:
            self.test_iterator = iter(tfds.as_numpy(self.ds_test))
            (x, y) = next(self.test_iterator)
        return process_cifar_batch(self.dshape, x, y)


    def next_downstream(self):
        try:
            (x, y) = next(self.downstream_iterator)
        except StopIteration:
            self.downstream_iterator = iter(tfds.as_numpy(self.ds_downstream))
            (x, y) = next(self.downstream_iterator)
        return process_cifar_batch(self.dshape, x, y)

    def get_batched_val(self):
        return self.batched_valset

    def get_batched_test(self):
        return self.batched_testset

    def batch_process_valtest(self, ds):
        XYs = [(X, Y) for (X, Y) in iter(tfds.as_numpy(ds))]
        Xs = [X[np.newaxis, ...] for (X, _) in XYs]
        Ys = [Y[np.newaxis, ...] for (_, Y) in XYs]

        # drop last batch if the size is irregular
        if Xs[-1].shape[1] != Xs[-2].shape[1]:
            Xs = Xs[:-1]
            Ys = Ys[:-1]
        Xs = np.concatenate(Xs, axis=0)
        Ys = np.concatenate(Ys, axis=0)
        return jax.vmap(process_cifar_batch, in_axes=(None, 0, 0))(self.dshape, Xs, Ys)


