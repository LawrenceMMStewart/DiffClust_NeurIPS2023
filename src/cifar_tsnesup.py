import numpy as np
import pickle, sys, os, json, argparse, ml_collections, math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import flax.linen as nn
import jax
import jax.numpy as jnp
from models import DCVanillaSup, DCBNTrainState, get_lpfuncs
from backbones import ResNet50
import utils
from flax.training import orbax_utils
import optax, orbax
from cifardata import CIFAR10ClustSupervised
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from absl import logging
from clu import metric_writers, parameter_overview
from typing import Callable, Dict
import tensorflow as tf
import tensorflow_datasets as tfds


"""
An example of how to load a model and create a tSNE visualization
of the embedding space.
"""

# if on GPU stop tensorflow from grabbing all memory
tf.config.set_visible_devices([], device_type='GPU')
logging.set_verbosity(logging.INFO)

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
def train_step_fn(state, X, Yhot, ncc, rngs):
    def forward(params, batch_stats, X, Yhot, ncc, rngs):
        (loss, outs), mutables = state.forward_fn({'params' : params, 'batch_stats' : batch_stats},
                                                   X, Yhot, ncc, rngs['noise'], True, True, rngs=rngs, mutable=['batch_stats'])
        return loss, (outs, mutables)
    (loss, (outs, mutables)), grads = jax.value_and_grad(forward, has_aux=True)(state.params, state.batch_stats, X, Yhot, ncc, rngs)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=mutables['batch_stats'])
    return state, outs, grads

@jax.jit
def eval_step_fn(state, X, Yhot, ncc, rngs):
    loss, outs= state.forward_fn({'params' : state.params, 'batch_stats' : state.batch_stats},
                                 X, Yhot, ncc, rngs['noise'], False, False, rngs=rngs)
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
    model = DCVanillaSup(
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


    for step in range(1, config.steps + 1):

        # update the rng for dropout and noise
        rngs = utils.fold_in_key(rngs, step, 'dropout')
        rngs = utils.fold_in_key(rngs, step, 'noise')
        rngs = utils.fold_in_key(rngs, step, 'augmentations')

        X, Yhot = handler.next_train(augment=True, key=rngs['augmentations'])
        state, outs, grads = train_step_fn(state, X, Yhot, 10, rngs)

        metrics.store(step, updates={k + '_train' : v.item() for (k, v) in outs.items()})


    # else checkpoint
        if step % config.eval_every == 0:
            outs = evaltest(state, handler, rngs, dset='val')

            improvements = metrics.update(step=step, updates={k + '_eval' : v.item() for (k, v) in outs.items()})
            metrics.write_scalar_values(step=step, writer=writer, save=config.save)
            for h in config.save_hooks:
                if improvements[h] and config.save:
                    save_model(step=0, name=h, config=config, state=state)
        if step == config.steps:
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
                name = f'embedding-{dset}__{h}.pickle'
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
                    prog='TSNE',
                    description='tSNE visualization of ResNet50 diffclust features')


    parser.add_argument('--emb_path', default='.', type=str, help='load path')

    args = parser.parse_args()
    config = ml_collections.ConfigDict(vars(args))

    with open(config.emb_path, 'rb') as f:
        X, Y = pickle.load(f)
    Yval = Y
    Xval = TSNE(n_components=2).fit_transform(X)

    TSNE_COLORS = np.array([
        '#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f',
        '#ff7f00','#cab2d6','#6a3d9a',
    ])


    num_unseen = 0
    num_digits = 10
    c_unseen = np.arange(num_unseen)
    c_seen = np.arange(num_unseen, num_digits)


    # plt.figure(figsize=(6.18, 6.18))
    for idx, c in enumerate(c_seen):
      c_indices = Yval.flatten() == c
      plt.plot(Xval[c_indices, 0], Xval[c_indices, 1],
               ls='',
               marker='o',
               ms=4,
               color=TSNE_COLORS[num_unseen + idx],
               alpha=0.5)

    ax = plt.gca()
    _ = plt.setp(ax.spines.values(), linewidth=1.5)
    _ = plt.xticks([])
    _ = plt.yticks([])
    _ = plt.savefig('valtsne_cifar10.pdf', bbox_inches='tight')
    plt.show()



