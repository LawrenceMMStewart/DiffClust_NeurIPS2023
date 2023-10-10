import flax.linen as nn
import flax, optax
from flax.training import train_state
import jax
import jax.numpy as jnp
from typing import Any, Callable, Sequence, Tuple, Union
from forests import pairwise_square_distance, Normal
import forests
import typing
from flax import core, struct
from typing import Type, Dict
from functools import partial


def get_lpfuncs(sigma : float = 0.1, num_samples : int = 1000,
                noise : Type[Normal] = Normal(), use_bias=True) -> dict:
    """
    Create dict consisting of functions to solve the spanning forest
    LP and constrained spanning forest LP, as well as their perturbed versions.
    Note any constrained function takes as input the target M (not the corresponding
    constraint matrix C).
    """

    funcs = {}

    # Forest LP and constrained LP.
    funcs['lp'] = lambda S, ncc : forests.mwst(S, ncc=ncc)
    funcs['clp'] = lambda S, Mconstr, ncc : forests.mwst_constrained(S, 2 * Mconstr - 1, ncc, use_bias=use_bias) # convert M to C

    # perturbed versions.
    pert_mwst = forests.make_perturbed_mwst(num_samples=num_samples, sigma=sigma, noise=noise)
    pert_mwst_constrained = forests.make_perturbed_mwst_constrained(num_samples=num_samples, sigma=sigma, noise=noise, use_bias=use_bias)
    funcs['pert_lp'] = lambda S, ncc, key : pert_mwst(S, ncc, key)
    funcs['pert_clp'] = lambda S, Mconstr, ncc, key : pert_mwst_constrained(S, 2 * Mconstr - 1, ncc, key)

    return funcs



class Baseline(nn.Module):
    '''
    baseline - i.e. backbone + linear layer
    for ce loss.
    '''
    backbone : nn.Module
    num_classes : int = 10

    @nn.compact
    def __call__(self, x, yhot, training=False):
        Z = self.embed(x, training=training)
        logits = nn.Dense(10, name='head')(Z)
        logprobs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        ce = -jnp.mean(jnp.sum(logprobs * yhot, axis=-1), axis=-1)
        error = jnp.mean(jnp.argmax(logits, axis=-1) != jnp.argmax(yhot, axis=-1), axis=-1)
        metrics = {'ce' : ce, 'err' : error}
        return ce, metrics

    def embed(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def eval_clust(self, x, yhot, training=False):
        Z = self.embed(x, training=training)
        M = yhot @ yhot.T
        S = - forests.pairwise_square_distance(Z)
        Ak, Mk = forests.mwst(S=S, ncc=self.num_classes)
        err = jnp.mean(jnp.sqrt((Mk - M) ** 2))
        return {'err' : err}



class BaselineTrainState(struct.PyTreeNode):
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  embed_fn : Callable = struct.field(pytree_node=False)
  clust_eval_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads, **kwargs):
    updates, new_opt_state = self.tx.update(
        grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, embed_fn, clust_eval_fn, params, tx, **kwargs):
    opt_state = tx.init(params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        embed_fn=partial(apply_fn, method=embed_fn),
        clust_eval_fn=partial(apply_fn, method=clust_eval_fn),
        params=params,
        tx=tx,
        opt_state=opt_state,
        **kwargs,
    )


class BaselineBNTrainState(struct.PyTreeNode):
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  embed_fn : Callable = struct.field(pytree_node=False)
  clust_eval_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  batch_stats : core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads, **kwargs):
    updates, new_opt_state = self.tx.update(
        grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, embed_fn, clust_eval_fn, params, batch_stats, tx, **kwargs):
    opt_state = tx.init(params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        embed_fn=partial(apply_fn, method=embed_fn),
        clust_eval_fn=partial(apply_fn, method=clust_eval_fn),
        params=params,
        batch_stats=batch_stats,
        tx=tx,
        opt_state=opt_state,
        **kwargs,
    )







class DCVanillaSup(nn.Module):
    '''
    supervised differentiable clustering
    '''
    backbone : nn.Module
    lpfuncs : Dict
    num_classes : int = 10
    sigma : float = 0.1 # noise
    num_samples : int = 1000 # number of monte carlo samples for estimator
    noise : Type[Normal] = Normal()
    exp_temp : float = 0 # if > 0 then take Sigma = exp ( t * S )
    correlate_noise : bool = True # if true then use same noise for both LP's in partial loss
    cosine_distance : bool = False # if true then normalize embeddings by their L2 norm, equivalent to cosine distance. 
    use_bias : bool = True # whether to use_bias in kruskals constrained version (wont change class as in lpfuncs, rather for reference).

    # call is equivalent to embedding the data
    @nn.compact
    def __call__(self, *args, **kwargs):
        Z = self.backbone(*args, **kwargs)
        if self.cosine_distance:
            Z = Z / jnp.linalg.norm(Z, axis=-1, keepdims=True)
        return Z

    def similarity(self, Z):
        S = - pairwise_square_distance(Z)
        if self.exp_temp > 0:
            S = jnp.exp(self.exp_temp * S)
        # standardizing reduces dependence of sigma on x
        S = (S - S.mean()) / S.std()
        return S

    def solve_lp(self, S, M, ncc, key, pert):
        if self.correlate_noise:
            key_unconstrained = key
            key_constrained = key
        else:
            key_constrained, key_unconstrained = jax.random.split(key)

        if pert:
            Ak, Fk, Mk = self.lpfuncs['pert_lp'](S, ncc, key_unconstrained)
            AkM, FkM, MkM = self.lpfuncs['pert_clp'](S, M, ncc, key_constrained)
        else:
            Ak, Mk = self.lpfuncs['lp'](S, ncc)
            AkM, MkM = self.lpfuncs['clp'](S, M, ncc)
            Fk = jnp.vdot(Ak, S)
            FkM = jnp.vdot(AkM, S)
        partial_loss = Fk - FkM
        return partial_loss, (Ak, Mk, Fk), (AkM, MkM, FkM)


    def forward(self, x, yhot, ncc, key, pert=True, training=True):

        Z = self.__call__(x, training=training)
        S = self.similarity(Z)
        M = yhot @ yhot.T
        partial_loss, (Ak, Mk, Fk), (AkM, MkM, FkM) = self.solve_lp(S, M, ncc, key, pert)
        l2_coincidence = jnp.mean(jnp.sqrt((Mk - M) ** 2))

        if pert:
            outs = {'pert_partial_loss' : partial_loss, 'pert_l2_coincidence' : l2_coincidence}
        else:
            outs = {'partial_loss' : partial_loss, 'l2_coincidence' : l2_coincidence}

        return partial_loss, outs

    def pert_and_unpert(self, x, yhot, ncc, key):

        """
        returns both perturbed and unperturbed metrics
        """
        Z = self.__call__(x, training=False)
        S = self.similarity(Z)

        M = yhot @ yhot.T

        pert_partial_loss, (pAk, pMk, pFk), (pAkM, pMkM, pFkM) = self.solve_lp(S, M, ncc, key, True)
        partial_loss, (Ak, Mk, Fk), (AkM, MkM, FkM) = self.solve_lp(S, M, ncc, key, False)
        l2_coincidence = jnp.mean(jnp.sqrt((Mk - M) ** 2))
        pert_l2_coincidence = jnp.mean(jnp.sqrt((pMk - M) ** 2))


        outs = {'pert_partial_loss' : pert_partial_loss,
                'pert_l2_coincidence' : pert_l2_coincidence,
                'partial_loss' : partial_loss,
                'l2_coincidence' : l2_coincidence
               }

        return outs





class DCVanillaSemiSup(nn.Module):
    '''
    semi-supervised differentiable clustering
    '''
    backbone : nn.Module
    lpfuncs : Dict
    num_classes : int = 10
    sigma : float = 0.1 # noise
    num_samples : int = 1000 # number of monte carlo samples for estimator
    noise : Type[Normal] = Normal()
    exp_temp : float = 0 # if > 0 then take Sigma = exp ( t * S )
    correlate_noise : bool = True # if true then use same noise for both LP's in partial loss
    cosine_distance : bool = False # if true then normalize embeddings by their L2 norm, equivalent to cosine distance. 
    use_bias : bool = True # whether to use_bias in kruskals constrained version (wont change class as in lpfuncs, rather for reference).

    @nn.compact
    def __call__(self, *args, **kwargs):
        Z = self.backbone(*args, **kwargs)
        if self.cosine_distance:
            Z = Z / jnp.linalg.norm(Z, axis=-1, keepdims=True)
        return Z

    def similarity(self, Z):
        S = - pairwise_square_distance(Z)
        if self.exp_temp > 0:
            S = jnp.exp(self.exp_temp * S)
        # standardizing reduces dependence of sigma on x
        S = (S - S.mean()) / S.std()
        return S

    def solve_lp(self, S, M, ncc, key, pert):
        if self.correlate_noise:
            key_unconstrained = key
            key_constrained = key
        else:
            key_constrained, key_unconstrained = jax.random.split(key)

        if pert:
            Ak, Fk, Mk = self.lpfuncs['pert_lp'](S, ncc, key_unconstrained)
            AkM, FkM, MkM = self.lpfuncs['pert_clp'](S, M, ncc, key_constrained)
        else:
            Ak, Mk = self.lpfuncs['lp'](S, ncc)
            AkM, MkM = self.lpfuncs['clp'](S, M, ncc)
            Fk = jnp.vdot(Ak, S)
            FkM = jnp.vdot(AkM, S)
        partial_loss = Fk - FkM
        return partial_loss, (Ak, Mk, Fk), (AkM, MkM, FkM)


    def mask_coincidence(self, yhot, batch_mask):
        Mtrue = yhot @ yhot.T
        # binary mask for coincidence: True --> masked, False --> information available 
        mask_mat = jnp.logical_or(batch_mask.reshape(1, -1), batch_mask.reshape(-1, 1))
        # need to correct diagonal to always be True (since even if label unobserved a point is always in its own cluster) use XOR
        mask_mat = jnp.logical_xor(mask_mat, jnp.diag(jnp.diag(mask_mat)))
        # Any values that are masked are set to 1/2.
        Mtarget = (1 - mask_mat) * Mtrue + 0.5 * mask_mat * jnp.ones_like(Mtrue)
        return Mtarget, Mtrue, mask_mat


    def forward(self, x, yhot, batch_mask, ncc, key, pert=True, training=True):

        Z = self.__call__(x, training=training)
        S = self.similarity(Z)
        Mtarget, Mtrue, mask_mat = self.mask_coincidence(yhot, batch_mask)
        partial_loss, (Ak, Mk, Fk), (AkM, MkM, FkM) = self.solve_lp(S, Mtarget, ncc, key, pert)
        l2_coincidence = jnp.mean(jnp.sqrt((Mk - Mtrue) ** 2))

        if pert:
            outs = {'pert_partial_loss' : partial_loss, 'pert_l2_coincidence' : l2_coincidence}
        else:
            outs = {'partial_loss' : partial_loss, 'l2_coincidence' : l2_coincidence}

        return partial_loss, outs

    def pert_and_unpert(self, x, yhot, ncc, key):

        """
        returns both perturbed and unperturbed metrics
        """
        Z = self.__call__(x, training=False)
        S = self.similarity(Z)

        M = yhot @ yhot.T

        pert_partial_loss, (pAk, pMk, pFk), (pAkM, pMkM, pFkM) = self.solve_lp(S, M, ncc, key, True)
        partial_loss, (Ak, Mk, Fk), (AkM, MkM, FkM) = self.solve_lp(S, M, ncc, key, False)
        l2_coincidence = jnp.mean(jnp.sqrt((Mk - M) ** 2))
        pert_l2_coincidence = jnp.mean(jnp.sqrt((pMk - M) ** 2))


        outs = {'pert_partial_loss' : pert_partial_loss,
                'pert_l2_coincidence' : pert_l2_coincidence,
                'partial_loss' : partial_loss,
                'l2_coincidence' : l2_coincidence
               }

        return outs



class DCTrainState(struct.PyTreeNode):
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  forward_fn : Callable = struct.field(pytree_node=False)
  eval_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads, **kwargs):
    updates, new_opt_state = self.tx.update(
        grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, forward_fn, eval_fn, params, tx, **kwargs):
    opt_state = tx.init(params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        forward_fn=partial(apply_fn, method=forward_fn),
        eval_fn=partial(apply_fn, method=eval_fn),
        params=params,
        tx=tx,
        opt_state=opt_state,
        **kwargs,
    )


class DCBNTrainState(struct.PyTreeNode):
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  forward_fn : Callable = struct.field(pytree_node=False)
  eval_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  batch_stats: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads, **kwargs):
    updates, new_opt_state = self.tx.update(
        grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, forward_fn, eval_fn, params, batch_stats, tx, **kwargs):
    opt_state = tx.init(params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        forward_fn=partial(apply_fn, method=forward_fn),
        eval_fn=partial(apply_fn, method=eval_fn),
        params=params,
        batch_stats=batch_stats,
        tx=tx,
        opt_state=opt_state,
        **kwargs,
    )


