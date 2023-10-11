import numpy as np
import pickle, sys, os, json, argparse, ml_collections, math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import flax.linen as nn
import jax
import jax.numpy as jnp
from models import DCVanillaSemiSup, DCBNTrainState, get_lpfuncs
from backbones import ResNet50
import utils
from flax.training import orbax_utils
import optax, orbax
from cifardata import CIFAR10ClustSemiSupervised
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from absl import logging
from clu import metric_writers, parameter_overview
from typing import Callable, Dict
import tensorflow as tf
import tensorflow_datasets as tfds

"""
Semi-Supervised differentible clustering on CIFAR 10
using a RESNET 50 architecture (modified for CIFAR).
"""


# if on GPU stop tensorflow from grabbing all memory
tf.config.set_visible_devices([], device_type='GPU')
logging.set_verbosity(logging.INFO)

def checkdir(config):
    if os.path.isdir(config.path) and config.save:
        print("Save dir already exists!!!")
        exit()
    elif config.save:
        os.mkdir(config.path)

def save_config(config : ml_collections.ConfigDict):
    with open(os.path.join(config.path, 'args.pkl'), 'wb') as f:
        pickle.dump(config.to_json(), f)
    with open(os.path.join(config.path, 'args.json'), 'w') as f:
        json.dump(config.to_json(), f, indent=4, sort_keys=True)


def get_optimizer(config : ml_collections.ConfigDict) -> optax.OptState:
    if config.optimizer == 'sgd':
        if config.momentum is None:
            optimizer = optax.sgd(config.lr, momentum=config.momentum, nesterov=config.nesterov)

        else:
            optimizer = optax.sgd(config.lr, momentum=float(config.momentum), nesterov=config.nesterov)
    elif config.optimizer == 'adam':
        optimizer = optax.adam(config.lr)
    elif config.optimizer == 'adamw':
        optimizer = optax.adamw(config.lr, weight_decay=config.weight_decay)
    return optimizer


def get_backbone(config : ml_collections.ConfigDict) -> nn.Module:
    if config.backbone == "ResNet50":
        return ResNet50(embedding_dim=config.embedding_dim)
    else:
        raise ValueError("Invalid backbone")


@jax.jit
def train_step_fn(state, X, Yhot, batch_mask, ncc, rngs):
    def forward(params, batch_stats, X, Yhot, batch_mask, ncc, rngs):
        (loss, outs), mutables = state.forward_fn({'params' : params, 'batch_stats' : batch_stats},
                                                   X, Yhot, batch_mask, ncc, rngs['noise'],
                                                  True, True, rngs=rngs, mutable=['batch_stats'])
        return loss, (outs, mutables)
    (loss, (outs, mutables)), grads = jax.value_and_grad(forward, has_aux=True)(state.params, state.batch_stats, X, Yhot,
                                                                                batch_mask, ncc, rngs)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=mutables['batch_stats'])
    return state, outs, grads

@jax.jit
def eval_step_fn(state, X, Yhot, ncc, rngs):
    batch_mask = jnp.full((X.shape[0], ), False) # all validation / test data is labeled.
    loss, outs= state.forward_fn({'params' : state.params, 'batch_stats' : state.batch_stats},
                                 X, Yhot, batch_mask, ncc, rngs['noise'], False, False, rngs=rngs)
    return outs

def evaltest(state, handler, rngs, dset='val'):
    if dset == 'val':
        X, Yhot = handler.get_batched_val()
    elif dset == 'test':
        X, Yhot = handler.get_batched_test()
    else:
        raise ValueError
    nbatches = X.shape[0]
    results = {}
    for (x, yhot) in zip(X, Yhot):
        outs = eval_step_fn(state, x, yhot, 10, rngs)
        if results == {}:
            results = outs
        else:
            results = jax.tree_map(lambda x, y : x + y, results, outs)

    return jax.tree_map(lambda x : x / nbatches, results)


def create_state(rngs, backbone=ResNet50(), optimizer=optax.sgd(1e-3), dshape=(32, 32, 3), sigma=0.1, num_samples=100,
                 cosine_distance=False, exp_temp=0, decorrelate_noise=False, bs=64, use_bias=True):

    lpfuncs =  get_lpfuncs(sigma=sigma, num_samples=num_samples, use_bias=use_bias)
    model = DCVanillaSemiSup(
        backbone=backbone,
        lpfuncs=lpfuncs,
        num_classes=10,
        sigma=sigma,
        num_samples=num_samples,
        exp_temp=exp_temp,
        correlate_noise=not decorrelate_noise,
        cosine_distance=cosine_distance,
        use_bias=use_bias)

    dummy_x = jnp.ones((bs,) + dshape)
    variables = model.init(rngs, dummy_x, training=False)

    print(parameter_overview.get_parameter_overview(variables['params']))
    state  = DCBNTrainState.create(apply_fn=model.apply,
                                 forward_fn=model.forward,
                                 eval_fn=model.pert_and_unpert,
                                 params=variables['params'],
                                 batch_stats=variables['batch_stats'],
                                 tx=optimizer)
    return state



def save_model(step: int,
               name : str,
               config : ml_collections.ConfigDict,
               state: DCBNTrainState):

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = {
        'state' : jax.tree_map(lambda x : x, state),
        'config' : dict(config),
        'step' : step
        }
        save_args = orbax_utils.save_args_from_target(ckpt)
        ckpt_path = os.path.join(config.path, name)
        orbax_checkpointer.save(ckpt_path, ckpt, save_args=save_args, force=True)


def load_model(config : ml_collections.ConfigDict, name : str, handler):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    load_path = os.path.join(config.path, name)

    rngs = {'params' : jax.random.PRNGKey(0)}
    opt = get_optimizer(config)
    backbone = get_backbone(config)
    dummy_state = create_state(rngs=rngs, backbone=backbone, optimizer=opt,
                               dshape=handler.dshape, bs=config.bs)
    dummy_ckpt = {
        'state' : dummy_state,
        'config' : dict(config),
        'step' : -1
        }
    ckpt_restored = orbax_checkpointer.restore(load_path, item=dummy_ckpt)
    return ckpt_restored



def plotmetrics(metrics : utils.MetricDict, save=False, path='./'):
    if save:
        N = len(metrics.metrics)
        M = math.ceil(math.sqrt(N)) # maximum size M x M grid
        L = math.ceil(N / M) # number of rows with M columns needed
        fig, ax = plt.subplots(L, M)
        for k, (key, metric) in enumerate(metrics.metrics.items()):
            i = k // M
            j = k % M
            if metric.type == 'AvgMetric':
                ax[i, j].plot(metric.steps, metric.vals, alpha=0.2)
                ax[i, j].plot(metric.avg_steps, metric.avg_vals, alpha=0.7)
            elif metric.type == 'Metric':
                label = f"Best = ({metric.best_step},{metric.best_val:.5f})"
                ax[i, j].plot(metric.steps, metric.vals, label=label, alpha=0.7)
                ax[i, j].legend(prop={'size': 6})
            ax[i, j].set_xlabel('steps', fontsize=6)
            ax[i, j].set_title(key, fontsize=9)

        fig.tight_layout(pad=2.0)
        plt.savefig(os.path.join(path, 'metrics.pdf'))





def train(config : ml_collections.ConfigDict, writer : metric_writers.SummaryWriter, handler):

    assert config.steps % config.eval_every == 0

    key = jax.random.PRNGKey(config.random_state)

    rng_keys = ['params', 'dropout', 'noise', 'mask', 'augmentations']
    rngs = { k : v for k, v in zip(rng_keys, jax.random.split(key, num=len(rng_keys)))}


    # intialize train_state
    opt = get_optimizer(config)
    backbone = get_backbone(config)
    # state = create_state(rngs=rngs, backbone=backbone, optimizer=opt, dshape=handler.dshape, bs=config.bs)
    state = create_state(rngs=rngs, backbone=backbone, optimizer=opt, dshape=handler.dshape, sigma=config.sigma,
                               num_samples=config.num_samples, exp_temp=config.exp_temp, cosine_distance=config.cosine_distance,
                               decorrelate_noise=config.decorrelate_noise, bs=config.bs, use_bias=config.use_bias)

    rngs = utils.fold_in_key(rngs, -1, 'params')

    metrics = utils.MetricDict({
        'pert_partial_loss_train' : {'type' : 'AvgMetric', 'better' : 'low'},
        'pert_l2_coincidence_train' : {'type' : 'AvgMetric', 'better' : 'low'},
        'partial_loss_eval' : {'type' : 'Metric', 'better' : 'low'},
        'l2_coincidence_eval' : {'type' : 'Metric', 'better' : 'low'}
    })

    config.save_hooks = ['l2_coincidence_eval']


    outs = evaltest(state, handler, rngs, dset='val')
    improvements = metrics.update(step=0, updates={k + '_eval' : v.item() for (k, v) in outs.items()})
    metrics.write_scalar_values(step=0, writer=writer, save=config.save)
    for h in config.save_hooks:
        if improvements[h] and config.save:
            save_model(step=0, name=h, config=config, state=state)

    config.stopped=-1
    stopping_count = 0
    max_count = config.early_stopping if config.early_stopping > 0 else np.inf

    for step in range(1, config.steps + 1):

        # update the rng for dropout and noise
        rngs = utils.fold_in_key(rngs, step, 'dropout')
        rngs = utils.fold_in_key(rngs, step, 'noise')
        rngs = utils.fold_in_key(rngs, step, 'augmentations')

        batch_mask, (X, Yhot) = handler.next_train(augment=True, key=rngs['augmentations'])
        state, outs, grads = train_step_fn(state, X, Yhot, batch_mask, 10, rngs)

        metrics.store(step, updates={k + '_train' : v.item() for (k, v) in outs.items()})


    # else checkpoint
        if step % config.eval_every == 0:
            outs = evaltest(state, handler, rngs, dset='val')

            improvements = metrics.update(step=step, updates={k + '_eval' : v.item() for (k, v) in outs.items()})
            metrics.write_scalar_values(step=step, writer=writer, save=config.save)
            for h in config.save_hooks:
                if improvements[h] and config.save:
                    save_model(step=0, name=h, config=config, state=state)

            if np.any([improvements[k] for k in improvements.keys() if k in config.save_hooks]):
                stopping_count = 0
            else:
                stopping_count +=1

        if stopping_count == max_count:
            print(f"Early stopping on step {step}")
            config.stopped=step
            save_model(step=step, name='last', config=config, state=state)
            return metrics, config

        elif step == config.steps:
            save_model(step=step, name='last', config=config, state=state)


    return metrics, config


def test(config, writer, handler):
    if config.save:
        key = jax.random.PRNGKey(config.random_state)
        rng_keys = ['params', 'dropout', 'noise', 'ds_shuffle', 'mask']
        rngs = { k : v for k, v in zip(rng_keys, jax.random.split(key, num=len(rng_keys)))}

        results = {}

        for h in config.save_hooks:
            ckpt = load_model(config, h, handler)
            state = ckpt['state']

            outs = evaltest(state, handler, rngs, dset='test')
            outs = {k : v.item() for (k, v) in outs.items()}

            print(f"Model chkpt {h} : {outs}")

            if config.save:
                result = {h.replace('_eval', '_test') : outs[h.replace('_eval', '')]}
                results = {**results, **result}
                writer.write_scalars(-1, result)

        with open(os.path.join(config.path, 'test_results.json'), 'w') as f:
            f.write(json.dumps(results, indent=4, sort_keys=True))



def embed_data(config : ml_collections.ConfigDict, handler):

    if config.save:

        key = jax.random.PRNGKey(config.random_state)

        rng_keys = ['params', 'dropout', 'noise', 'ds_shuffle', 'mask']
        rngs = { k : v for k, v in zip(rng_keys, jax.random.split(key, num=len(rng_keys)))}

        for h in config.save_hooks:

            ckpt = load_model(config, h, handler)
            state = ckpt['state']

            for dset in ['downstream', 'val', 'test']:

                dset_iterator = iter(tfds.as_numpy(getattr(handler, f'ds_{dset}')))
                next_method = getattr(handler, f'next_{dset}')
                X, Yhot = next_method()
                Y = jnp.argmax(Yhot, axis=-1, keepdims=True)
                V = state.apply_fn({'params' : state.params, 'batch_stats' : state.batch_stats}, X, training=False, rngs=rngs)
                name = f'embedding-{dset}_{h}.pickle'
                with open(os.path.join(config.path, name), 'wb') as f:
                    pickle.dump((V, Y), f)


def visualize_embeddings(config : ml_collections.ConfigDict, handler):

    key = jax.random.PRNGKey(0)

    if config.save:
        for hook in config.save_hooks:

            ckpt = load_model(config, hook, handler)
            state = ckpt['state']
            rng_keys = ['params', 'dropout', 'noise', 'ds_shuffle', 'mask']
            rngs = { k : v for k, v in zip(rng_keys, jax.random.split(key, num=len(rng_keys)))}

            X_val, Yhot_val = handler.next_val()
            Y_val = jnp.argmax(Yhot_val, axis=-1, keepdims=True)
            # embed train and val data
            V_val = state.apply_fn({'params' : state.params, 'batch_stats' : state.batch_stats}, X_val, training=False, rngs=rngs)

            print("performing PCA on val set")
            V_val_pca = PCA(n_components=2).fit_transform(V_val)
            print("performing TSNE on val set")
            V_val_tsne = TSNE(n_components=2).fit_transform(V_val)


            cmap = cm.get_cmap('viridis', 10)
            color_map = np.array([cmap(i) for i in range(10)])
            fig, ax = plt.subplots(1, 2)
            ax[0].scatter(V_val_pca[:, 0], V_val_pca[:, 1], color=color_map[Y_val], marker='.', alpha=0.4, s=10)
            ax[0].set_title("PCA val set ")
            ax[1].scatter(V_val_tsne[:, 0], V_val_tsne[:, 1], color=color_map[Y_val], marker='.', alpha=0.4, s=10)
            ax[1].set_title("TSNE val set")
            fig.tight_layout()
            fig.savefig(os.path.join(config.path, f"vis_{hook}.pdf"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Semisup Diffclust',
                    description='Train ResNet50 in semi-supervised setting on CIFAR10 using diffclust')


    parser.add_argument('--random_state', default=0, type=int, help='random seed')

    parser.add_argument('--dataset', default='mnist', type=str, help='mnist, fashionmnist')

    parser.add_argument('--ncc', default=10, type=int, help='number of clusters / connected components in kruskals')

    parser.add_argument('--sigma', default=0.1, type=float, help='noise level for perturbations')

    parser.add_argument('--num_samples', default=1000, type=int, help='number of samples in monte carlo estimate of jacobian')

    parser.add_argument('--steps', default=50000, type=int, help='number of training steps')

    parser.add_argument('--eval_every', default=100, type=int, help='number of training steps')

    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer to use')

    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')

    parser.add_argument('--exp_temp', default=0, type=float, help='If exp_temp>0, then use exp( t * S) as similarity matrix')

    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (if optimnizer includes it)')

    parser.add_argument('--momentum', default=None, help='momentum for SGD')

    parser.add_argument('--dropout', default=0., type=float, help='dropout value if model uses dropout')

    parser.add_argument('--nesterov', default=False, action='store_true', help='use nesterov momentum for SGD')

    parser.add_argument('--reuse_noise', action="store_true", default=False, help='If True, the use the same noise sample Z for each train step')

    parser.add_argument('--decorrelate_noise', action="store_true", default=False, help='If True, use different noise for LP and constrained LP.')

    parser.add_argument('--save', action='store_true', default=False, help='save best train states')

    parser.add_argument('--path', type=str, default='ss_mnist/exp', help='name for save dir')

    parser.add_argument('--backbone', default='ResNet50', type=str, help='backbone for diff clust model')

    parser.add_argument('--embedding_dim', default=256, type=int, help="Dimension of the embedding space.")

    parser.add_argument('--cosine_distance', default=False, action='store_true', help="If True project features onto unit ball.")

    parser.add_argument('--use_bias', default=True, type=bool, help="Whether or not to use bias in the constrained kruskal's")

    parser.add_argument('--reshuffle', default=True, type=bool, help="keep the same batches at each epoch, do not reshuffle data")

    parser.add_argument('--bs', default=64, type=int, help='Batch size to use for clustering')

    parser.add_argument('--testval_bs', default=64, type=int, help='Batch size for the validation and test data (no affect on classif).')

    parser.add_argument('--N_labeled', default=1000, type=int, help='Number of labeled data points')

    parser.add_argument('--lbs', default=32, type=int, help='Number of points in batch to have labels')

    parser.add_argument('--K', default=0, type=int, help='Number classes to completely drop labels from')

    parser.add_argument('--early_stopping', default=-1, type=int, help='If positive, stop after this number of non-hook improvements.')


    args = parser.parse_args()
    config = ml_collections.ConfigDict(vars(args))
    np.random.seed(config.random_state); tf.random.set_seed(config.random_state)

    checkdir(config)
    writer = metric_writers.create_default_writer(logdir=config.path) if config.save else None

    # load dataset
    handler = CIFAR10ClustSemiSupervised(bs=config.bs,
                                         testval_bs=config.testval_bs,
                                         reshuffle=config.reshuffle,
                                         backbone=config.backbone,
                                         N_labeled=config.N_labeled,
                                         lbs=config.lbs,
                                         K=config.K)



    # training
    metrics, config = train(config, writer, handler)
    plotmetrics(metrics, save=config.save, path=config.path)
    save_config(config)
    test(config, writer, handler)
    embed_data(config, handler)
    visualize_embeddings(config, handler)
    if writer is not None:
        writer.close()
