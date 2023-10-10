import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
np.random.seed(0)

# if on GPU stop tensorflow from grabbing all memory
tf.config.set_visible_devices([], device_type='GPU')

# functions that handle data ids
def group_ids(ids_dict):
    return jnp.concatenate([v for v in ids_dict.values()])


def balanced_split(key, ids_dict, N_total):
    purged = { k : v for (k, v) in ids_dict.items() if len(v) > 0}
    N_classes = len(purged.keys())
    # number of labels per class
    Npc = N_total // N_classes
    keys = jax.random.split(key, N_classes)

    split = { k : jax.random.choice(keys[k], v, shape=(Npc, ), replace=False) for (k, v) in purged.items()}
    remains = {k : jnp.setdiff1d(purged[k], split[k]) for k in purged.keys()}
    return split, remains


def fuse_dicts(dict1, dict2):
    fused = {}
    d1k = [k for k in dict1.keys()]
    d2k = [k for k in dict2.keys()]

    all_keys = np.union1d(d1k, d2k)
    for k in all_keys:
        if k in dict1.keys() and k in dict2.keys():
            fused[k] = jnp.concatenate((dict1[k], dict2[k]))
        elif k in dict1.keys() and k not in dict2.keys():
            fused[k] = jnp.copy(dict1[k])
        elif k not in dict1.keys() and k in dict2.keys():
            fused[k] = jnp.copy(dict2[k])
        else:
            print("Something has gone wrong")
            exit()
    return fused


def discard_lastK(ids_dict, K=0, num_classes=10):
    discarded = {k : ids_dict[k] for k in range(10) if k >= num_classes - K}
    kept = {k : ids_dict[k] for k in range(10) if k < num_classes - K}
    return discarded, kept



class SS_CIFAR10_Loader():

    def __init__(self, key=jax.random.PRNGKey(0)):
        self.key = key

        X, Y= tfds.load(name='cifar10', batch_size=-1, data_dir='./tfds', split='train', as_supervised=True)
        X = np.array(X)
        Y = np.array(Y)

        label_ids = {y : jnp.where(Y == y)[0] for y in range(10)}


        downstream_ids, train_ids = balanced_split(self.key, label_ids, 1000)
        self.downstream_ids = downstream_ids
        self.train_ids = train_ids


    def get_testset(self, bs=5000):
        """
        5k test set (coming from first 5k points of classic cifar10 test split)
        """
        ds = tfds.load(name='cifar10', batch_size=-1, data_dir='./tfds', split='test', as_supervised=True)
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.take(5000)
        if bs == -1:
            bs = 5000
        return ds.batch(bs)


    def get_valset(self, bs=5000):
        """
        5k val set (coming from second 5k points of classic cifar10 test split)
        """
        ds = tfds.load(name='cifar10', batch_size=-1, data_dir='./tfds', split='test', as_supervised=True)
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.skip(5000)
        if bs == -1:
            bs = 5000
        return ds.batch(bs)


    def get_downstream(self, bs=-1, reshuffle=True):

        X, Y= tfds.load(name='cifar10', batch_size=-1, data_dir='./tfds', split='train', as_supervised=True)
        X = np.array(X)
        Y = np.array(Y)

        keep = group_ids(self.downstream_ids)
        X = X[keep]
        Y = Y[keep]
        bs = 1000 if bs == -1 else bs

        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        ds = ds.shuffle(buffer_size = 1000, reshuffle_each_iteration=reshuffle, seed=0)
        ds = ds.batch(bs)
        return ds


    def get_train_datasets(self, K=0, lbs=-1, ubs=-1, N_labeled=5000, reshuffle=True):

        X, Y= tfds.load(name='cifar10', batch_size=-1, data_dir='./tfds', split='train', as_supervised=True)
        X = np.array(X)
        Y = np.array(Y)

        # 1000 points are kept aside to make up downstream data sets 
        keys = jax.random.split(self.key, 2)

        tids = self.train_ids

        # remove labels for last K digits 
        lastK_ids, firstK_ids = discard_lastK(tids, K=K) # ids to become unlabeled and stay labeled

        lids, lids_rest = balanced_split(keys[1], firstK_ids, N_labeled) # subsample the labeled
        uids = fuse_dicts(lastK_ids, lids_rest) # fuse the unlabeled and non-sampled to form unlabeled


        lids = group_ids(lids)
        uids = group_ids(uids)
        assert len(np.intersect1d(lids, uids)) == 0
        assert len(lids) + len(uids) == 49000

        Xl, Yl = (X[lids], Y[lids])
        Xu, Yu = (X[uids], Y[uids])

        X_train = jnp.concatenate((Xl, Xu), axis=0)
        Y_train = jnp.concatenate((Yl, Yu), axis=0)

        ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).enumerate()
        lds = ds.take(len(Xl)) # cannot do take(N_labeled) in case there are a few less due to K!!
        uds = ds.skip(len(Xl))

        lds = lds.shuffle(buffer_size = len(Xl), reshuffle_each_iteration=reshuffle, seed=0)
        uds = uds.shuffle(buffer_size = len(Xu), reshuffle_each_iteration=reshuffle, seed=0)

        if lbs == -1:
            lbs = len(Xl)
        if ubs == -1:
            ubs = len(Xu)

        lds = lds.batch(lbs)
        uds = uds.batch(ubs)

        # 1 indicates label missing
        mask = jnp.concatenate((
            jnp.zeros((len(Xl), ), dtype=bool),
            jnp.ones((len(Xu), ), dtype=bool)
        ))

        return lds, uds, mask



class FS_CIFAR10_Loader():
    '''
    fully supervised nist loader : cifar10, fashion mnist.
    '''

    def __init__(self, dataset='cifar10', key=jax.random.PRNGKey(0)):
        self.key = key

        X, Y= tfds.load(name='cifar10', batch_size=-1, data_dir='./tfds', split='train', as_supervised=True)
        X = np.array(X)
        Y = np.array(Y)

        label_ids = {y : jnp.where(Y == y)[0] for y in range(10)}


        downstream_ids, train_ids = balanced_split(self.key, label_ids, 1000)
        self.downstream_ids = downstream_ids
        self.train_ids = train_ids


    def get_testset(self, bs=5000):
        """
        5k test set (coming from first 5k points of classic cifar10 test split)
        """
        ds = tfds.load(name='cifar10', batch_size=-1, data_dir='./tfds', split='test', as_supervised=True)
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.take(5000)
        if bs == -1:
            bs = 5000
        return ds.batch(bs)


    def get_valset(self, bs=5000):
        """
        5k val set (coming from second 5k points of classic cifar10 test split)
        """
        ds = tfds.load(name='cifar10', batch_size=-1, data_dir='./tfds', split='test', as_supervised=True)
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.skip(5000)
        if bs == -1:
            bs = 5000
        return ds.batch(bs)


    def get_downstream(self, bs=-1, reshuffle=True):

        X, Y= tfds.load(name='cifar10', batch_size=-1, data_dir='./tfds', split='train', as_supervised=True)
        X = np.array(X)
        Y = np.array(Y)

        keep = group_ids(self.downstream_ids)
        X = X[keep]
        Y = Y[keep]
        bs = 1000 if bs == -1 else bs

        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        ds = ds.shuffle(buffer_size = 1000, reshuffle_each_iteration=reshuffle, seed=0)
        ds = ds.batch(bs)
        return ds


    def get_train_dataset(self, bs=-1, reshuffle=True):

        X, Y= tfds.load(name='cifar10', batch_size=-1, data_dir='./tfds', split='train', as_supervised=True)
        X = np.array(X)
        Y = np.array(Y)

        # 1000 points are kept aside to make up downstream data sets 
        keys = jax.random.split(self.key, 2)

        tids = group_ids(self.train_ids)

        X_train, Y_train = (X[tids], Y[tids])

        if bs==-1:
            bs = len(X_train)

        ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        ds = ds.shuffle(buffer_size = len(X_train), reshuffle_each_iteration=reshuffle, seed=0)
        ds = ds.batch(bs)
        return ds






