import nistloader
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from dataclasses import dataclass
import tensorflow as tf
from typing import Union
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], device_type='GPU')


'''
Data set dataclasses for MNIST and Fashion MNIST
in the fully and semi-supervised settings
'''

@partial(jax.jit, static_argnums=(0))
def process_nist_batch(dshape : tuple, x, y):
    x = x.reshape((len(x), ) + dshape)
    x = x / 255.
    yhot = jax.nn.one_hot(y, 10)
    return x, yhot


@partial(jax.jit, static_argnums=(0))
def process_nist_masked_batch(dshape : tuple, x, y, batch_ids, ds_mask):
    x = x.reshape((len(x),) + dshape)
    x = x / 255.
    yhot = jax.nn.one_hot(y, 10)
    batch_mask = ds_mask[batch_ids]
    return batch_mask, (x, yhot)


@jax.jit
def fuse(Xl, Yl, idsl, Xu, Yu, idsu):
    X = jnp.concatenate((Xl, Xu), axis=0)
    Y = jnp.concatenate((Yl, Yu), axis=0)
    ids = jnp.concatenate((idsl, idsu), axis=0)
    return ids, (X, Y)


@dataclass
class NISTClustSupervised:

    dataset : str = 'mnist'
    bs : int = 64
    testval_bs : int = 64
    reshuffle : bool = True
    backbone : str = 'CNN'
    only_downstream : bool = False

    def __post_init__(self):
        super().__init__()

        if self.backbone == "CNN":
            self.dshape = (28, 28, 1)
        elif self.backbone == "MLP":
            self.dshape = (784, )
        else:
            raise ValueError

        if self.dataset == 'mnist':
            loader = nistloader.FS_MNIST_Loader()
        elif self.dataset == 'fashionmnist':
            loader = nistloader.FS_Fashion_Loader()

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


    def next_train(self):

        try:
            (x, y) = next(self.train_iterator)
            if x.shape[0] != self.bs:
                self.train_iterator = iter(tfds.as_numpy(self.ds_train))
                (x, y) = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(tfds.as_numpy(self.ds_train))
            (x, y) = next(self.train_iterator)

        return process_nist_batch(self.dshape, x, y)

    def next_val(self):
        try:
            (x, y) = next(self.val_iterator)
        except StopIteration:
            self.val_iterator = iter(tfds.as_numpy(self.ds_val))
            (x, y) = next(self.val_iterator)
        return process_nist_batch(self.dshape, x, y)


    def next_test(self):
        try:
            (x, y) = next(self.test_iterator)
        except StopIteration:
            self.test_iterator = iter(tfds.as_numpy(self.ds_test))
            (x, y) = next(self.test_iterator)
        return process_nist_batch(self.dshape, x, y)


    def next_downstream(self):
        try:
            (x, y) = next(self.downstream_iterator)
        except StopIteration:
            self.downstream_iterator = iter(tfds.as_numpy(self.ds_downstream))
            (x, y) = next(self.downstream_iterator)
        return process_nist_batch(self.dshape, x, y)

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
        return jax.vmap(process_nist_batch, in_axes=(None, 0, 0))(self.dshape, Xs, Ys)





@dataclass
class NISTClustSemiSupervised:

    dataset : str = 'mnist'
    bs : int = 64
    testval_bs : int = 64
    reshuffle : bool = True
    backbone : str = 'CNN'
    only_downstream : bool = False
    N_labeled : int = 100
    lbs : int = 32
    K : int = 0
    ubs : Union[int, None]  = None

    def __post_init__(self):
        super().__init__()

        assert self.lbs <= self.bs
        if self.ubs is None:
            self.ubs = self.bs - self.lbs


        if self.backbone == "CNN":
            self.dshape = (28, 28, 1)
        elif self.backbone == "MLP":
            self.dshape = (784, )

        if self.dataset == 'mnist':
            loader = nistloader.SS_MNIST_Loader()
        elif self.dataset == 'fashionmnist':
            loader = nistloader.SS_Fashion_Loader()

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


    def next_train(self):
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
        return process_nist_masked_batch(self.dshape, x, y, ids, self.mask)



    def next_labeled(self):
        try:
            idsl, (xl, yl) = next(self.lds_iterator)
        except StopIteration:
            self.lds_iterator = iter(tfds.as_numpy(self.lds))
            idsl, (xl, yl) = next(self.lds_iterator)
        x, yhot = process_nist_batch(self.dshape, xl, yl)
        return x, yhot


    def next_val(self):
        try:
            (x, y) = next(self.val_iterator)
        except StopIteration:
            self.val_iterator = iter(tfds.as_numpy(self.ds_val))
            (x, y) = next(self.val_iterator)
        return process_nist_batch(self.dshape, x, y)


    def next_test(self):
        try:
            (x, y) = next(self.test_iterator)
        except StopIteration:
            self.test_iterator = iter(tfds.as_numpy(self.ds_test))
            (x, y) = next(self.test_iterator)
        return process_nist_batch(self.dshape, x, y)


    def next_downstream(self):
        try:
            (x, y) = next(self.downstream_iterator)
        except StopIteration:
            self.downstream_iterator = iter(tfds.as_numpy(self.ds_downstream))
            (x, y) = next(self.downstream_iterator)
        return process_nist_batch(self.dshape, x, y)

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
        return jax.vmap(process_nist_batch, in_axes=(None, 0, 0))(self.dshape, Xs, Ys)


