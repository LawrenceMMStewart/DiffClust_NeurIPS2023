import numpy as np
import jax.numpy as jnp
from jax import jit, grad
import jax
from jax.lax import while_loop, cond
from typing import Tuple, Callable


@jit
def pairwise_square_distance(X):
    """
    euclidean pairwise square distance between data points
    """
    n = X.shape[0]
    G = jnp.dot(X, X.T)
    g = jnp.diag(G).reshape(n, 1)
    o = jnp.ones_like(g)
    return jnp.dot(g, o.T) + jnp.dot(o, g.T) - 2 * G


def kruskals_step(t, A, M, C_star, triu_ids, nedges):
    """
    Performs a single step of Kruskals algorithm for a complete graph
    with n verticies, with a constraint matrix C_star (assumed to be valid).

    Inputs:
        t: int (algo time step)
        A: array [n x n], current adjacency matrix of forest
        M: array [n x n], cluster connectivity matrix / coincidence matrix
        C_star: array [n x n], constraint matrix -1 = must not link, 0 = unconstrained, 1 = must link
        triu_ids : array [n(n-1)/2, ], ids of upper triangular of similarity matrix
        nedges : int, number of edges currently in forest.
    Outputs:
        A, M, nedges.
    """
    n = M.shape[0]
    triu_rows, triu_cols = triu_ids
    i, j = triu_rows[t], triu_cols[t]
    I = jnp.eye(n)
    # U is the ids of points in either the same cluster as nodes i and j
    U = M[i] + M[j]
    # J is the vector with ones at point i and j
    J = I[i] + I[j]

    # boolean True if adding edge does not lead to a cycle
    acyclic = M[i, j] == 0

    #  M_tentative is the new M if we were to make the update
    M_tentative = M +  U.reshape(n, 1) * (jnp.outer(U, U) - M)
    # cannot_update is a boolean, return True if C_star forbids any part of the new update
    cannot_update =  (M_tentative * (C_star == -1)).any()

    a = acyclic * (1 - cannot_update)

    # update M
    M = (1 - a) * M + a * M_tentative
    # M = M + a * U.reshape(n, 1) * (jnp.outer(U, U) - M)

    # update A 
    A = A + a * (jnp.outer(J, J) - I * J.reshape(n, 1))
    nedges = nedges + a
    return A, M, nedges


def kruskals_loop_body(vals):
    """
    Collects parameters of kruskals_step as a dictionary,
    used in kruskals_constrained.

    Inputs:
        vals : dict
    Outputs:
        vals : dict
    """
    nedges, nedges_prev = vals['nedges'], vals['nedges_prev']
    A, M, nedges_new = kruskals_step(vals['t'], vals['A'],
                                     vals['M'], vals['C_star'],
                                     vals['triu_ids'],
                                     vals['nedges'])

    vals['A'] = A
    vals['M'] = M
    # vals['Ahist'].append(A)
    # vals['Mhist'].append(M)
    vals['t']+=1
    vals['nconnected'] -= (nedges_new - nedges)
    vals['nedges'] = nedges_new
    vals['nedges_prev'] = nedges

    return vals


@jit
def kruskals_constrained(D, C_star, ncc=1, use_bias=True):
    """
    Runs kruskals algorithm respecting a constraint matrix (assumed to be valid).


    Inputs:
        D : array [n x n], symmetric pair-wise distance matrix.
        C_star : array [n x n], constraint matrix -1 = must not link, 0 = unconstrained, 1 = must link
        ncc : int, number of connected components upon which to terminate algorithm.
        use_bias : bool, if true then add all "must connect" edges to the tree straight away", other-wise
                    add them when they would normally appear in the algorithm.
    Outputs:
        A : array [n x n], adjacency matrix of forest.
        M : array [n x n], cluster connectivity matrix / coincidence matrix.

    """

    n = len(D)
    nconnected = jnp.array(n)
    t = jnp.array(0)
    # indicies for upper triangular (excluding diagonal of Distance Matrix)
    triu_ids = jnp.triu_indices(n, k=1)
    triu_rows, triu_cols = triu_ids

    biased = lambda D, C_star : D - (C_star * (D.max() - D.min()))
    unbiased = lambda D, C_star : D

    D_constrained = cond(use_bias, biased, unbiased, D, C_star)


    # sort by distance
    perm = jnp.argsort(D_constrained[triu_ids])
    # (rows, cols) of upper triangular are sorted by distance
    triu_rows = triu_rows[perm]
    triu_cols = triu_cols[perm]
    triu_ids = (triu_rows, triu_cols)

    A = jnp.eye(n)
    M = jnp.eye(n)

    nedges = jnp.array(0)
    nedges_prev = jnp.array(0)

    vals = {
        'max_steps' : len(triu_rows),
        'A' : A,
        'M' : M,
        't' : t,
        'C_star' : C_star,
        'nconnected' : nconnected,
        'triu_ids' : triu_ids,
        'nedges' : nedges,
        'nedges_prev' : nedges_prev
    }

    def cond_fn(vals):
        cond1 = vals['nconnected'] != ncc
        cond2 = vals['t'] <= vals['max_steps']
        return jnp.logical_and(cond1, cond2)

    vals = while_loop(cond_fn, kruskals_loop_body, vals)

    return vals['A'], vals['M']



@jit
def kruskals(D, ncc=1):
    """
    Kruskals algorithm unconstrained.

    Inputs:
        D : array [n x n], symmetric pair-wise distance matrix.
        ncc : int, number of connected components upon which to terminate algorithm.

    Outputs:
        A : array [n x n], adjacency matrix of forest.
        M : array [n x n], cluster connectivity matrix / coincidence matrix.
    """
    C_star = jnp.zeros_like(D)
    return kruskals_constrained(D, C_star, ncc=ncc)

@jit
def mwst(S, ncc=1):
    """
    Same as kruskals, but for similarity matrix S.
    """
    return kruskals(-S, ncc=ncc)

@jit
def mwst_constrained(S, C_star, ncc=1, use_bias=True):
    """
    Same as kruskals_constrained, but for similarity matrix S.
    """
    return kruskals_constrained(-S, C_star, ncc=ncc, use_bias=use_bias)


class Normal:
  """Normal distribution."""

  def sample(self,
             seed: jax.random.PRNGKey,
             sample_shape: Tuple[int]) -> jax.Array:
    return jax.random.normal(seed, sample_shape)

  def log_prob(self, inputs: jax.Array) -> jax.Array:
    return -0.5 * inputs ** 2


def make_perturbed_mwst(num_samples: int = 1000, sigma: float = 0.1, noise=Normal()) -> Callable:
    """
    Returns a perturbed version of mwst, taking the same arguments in addition to a
    PRNGKey for noise.

    Inputs:
        num_samples : number of samples for monte carlo estimate of expectation.
        sigma : noise amplitude perturbations.
        noise : A class following the structure given of the code for Normal.

    Returns:
        forward_pert : Callable with custom JVP.
    """


    @jax.custom_jvp
    def forward_pert(S, ncc, rng):
        samples = noise.sample(seed=rng,
                               sample_shape=(num_samples,) + S.shape)

        Ak_z, Mk_z = jax.vmap(mwst, in_axes=(0, None))(S + sigma * samples, ncc)

        # perturbed argmax and its corresponding coincidence / cluster connectivity matrix
        Akeps = jnp.mean(Ak_z, axis=0)
        Mkeps = jnp.mean(Mk_z, axis=0)

        # perturbed mwst value
        max_values = jnp.einsum('nd,nd->n',
                                jnp.reshape(S + sigma * samples,
                                            (num_samples, -1)),
                                jnp.reshape(Ak_z, (num_samples, -1)))
        Fkeps = jnp.mean(max_values)

        return Akeps, Fkeps, Mkeps


    def pert_jvp(tangent, _, S, ncc, rng):

        samples = noise.sample(seed=rng,
                               sample_shape=(num_samples,) + S.shape)


        Ak_z, Mk_z = jax.vmap(mwst, in_axes=(0, None))(S + sigma * samples, ncc)


        #argmax jvp
        nabla_z_flat = -jax.vmap(jax.grad(noise.log_prob))(samples.reshape([-1]))
        tangent_flat = 1.0 / (num_samples * sigma) * jnp.einsum(
            'nd,ne,e->d',
            jnp.reshape(Ak_z, (num_samples, -1)),
            jnp.reshape(nabla_z_flat, (num_samples, -1)),
            jnp.reshape(tangent, (-1,)))
        jvp_argmax = jnp.reshape(tangent_flat, S.shape)


        # max jvp
        pert_argmax = jnp.mean(Ak_z, axis=0)
        jvp_max = jnp.sum(pert_argmax * tangent)
        return jvp_argmax, jvp_max, jnp.zeros_like(tangent) # last output is dummy since we never do grad w.r.t M


    forward_pert.defjvps(pert_jvp, None, None)

    return forward_pert



def make_perturbed_mwst_constrained(num_samples: int = 1000, sigma: float = 0.1, noise=Normal(), use_bias : bool = True) -> Callable:
    """
    Returns a perturbed version of mwst_constrained, taking the same arguments in addition to a
    PRNGKey for noise.

    Inputs:
        num_samples : number of samples for monte carlo estimate of expectation.
        sigma : noise amplitude perturbations.
        noise : A class following the structure given of the code for Normal.
        use_bias : Use the bias method in constrained kruskal's.

    Returns:
        forward_pert : Callable with custom JVP.
    """

    @jax.custom_jvp
    def forward_pert(S, C_star, ncc, rng):
        samples = noise.sample(seed=rng, sample_shape=(num_samples,) + S.shape)
        AkM_z, MkM_z = jax.vmap(mwst_constrained, in_axes=(0, None, None, None))(S + sigma * samples, C_star, ncc, use_bias)

        AkMeps = jnp.mean(AkM_z, axis=0)
        MkMeps = jnp.mean(MkM_z, axis=0)

        max_values = jnp.einsum('nd,nd->n',
                                jnp.reshape(S + sigma * samples,
                                            (num_samples, -1)),
                                jnp.reshape(AkM_z, (num_samples, -1)))

        FkMeps = jnp.mean(max_values)
        return AkMeps, FkMeps, MkMeps


    def pert_jvp(tangent, _, S, C_star, ncc, rng):
        samples = noise.sample(seed=rng, sample_shape=(num_samples,) + S.shape)
        AkM_z, MkM_z = jax.vmap(mwst_constrained, in_axes=(0, None, None, None))(S + sigma * samples, C_star, ncc, use_bias)
        # argmax jvp
        nabla_z_flat = -jax.vmap(jax.grad(noise.log_prob))(samples.reshape([-1]))
        tangent_flat = 1.0 / (num_samples * sigma) * jnp.einsum('nd,ne,e->d',
                                                                jnp.reshape(AkM_z, (num_samples, -1)),
                                                                jnp.reshape(nabla_z_flat, (num_samples, -1)),
                                                                jnp.reshape(tangent, (-1,)))

        jvp_argmax = jnp.reshape(tangent_flat, S.shape)

        # max jvp
        pert_argmax = jnp.mean(AkM_z, axis=0)
        jvp_max = jnp.sum(pert_argmax * tangent)
        return jvp_argmax, jvp_max, jnp.zeros_like(tangent) # third is dummy for Mk


    forward_pert.defjvps(pert_jvp, None, None, None)

    return forward_pert



def argsort_first_k(x : jnp.ndarray, k : int) -> jnp.ndarray:
    '''
    Inputs:
        x : array, to be sorted.
        k : int, number of smallest elements to find.
    Outputs:
        ids : array of size k of indicies of smallest k elements
    '''

    first_k_ids = jnp.argpartition(x, k)[:k]
    sorted_ids_of_ids = jnp.argsort(jnp.take(x, first_k_ids))
    return first_k_ids[sorted_ids_of_ids]



def prims_adjacency_update(vals, _):
    A = vals['A']
    M = vals['M']
    I = vals['I']
    D = vals['D']
    ids = vals['ids']
    n = A.shape[0]

    in_forest = M[:, 0] == 1
    out_forest = M[:, 0] == 0

    mask = jnp.outer(in_forest, in_forest) + jnp.outer(out_forest, out_forest) + jnp.eye(n)
    D_masked = D + mask * D.max() * 2

    i, j = jnp.unravel_index(jnp.argmin(D_masked), D_masked.shape)

    J = (I[i] + I[j]).reshape(n, 1)
    U = (M[i] + M[j]).reshape(n, 1)

    A = A + jnp.outer(J, J) - jnp.eye(n) * J
    M = M + (jnp.outer(U, U) - M) * U

    vals['A'] = A
    vals['M'] = M

    return vals, _


def prims_adjacency(D : jnp.ndarray) -> jnp.ndarray:
    n = D.shape[0]
    A = jnp.eye(n)
    M = jnp.eye(n)
    I = jnp.eye(n)
    ids = jnp.arange(A.shape[0])
    vals = {'A' : A, 'I': I, 'D' : D, 'ids' : ids, 'M' : M}

    vals, _  = jax.lax.scan(f=prims_adjacency_update, init=vals, xs=None, length=n-1)
    return vals['A']


def pfkb_step(vals : dict, _ ) -> dict:
    """
    prims forward, kruskals back.
    """
    I = vals['I']
    A = vals['A']
    M = vals['M']
    iinds, jinds = vals['inds']
    count = vals['count']
    n = A.shape[0]

    i, j = (iinds[count], jinds[count])

    J = (I[i] + I[j]).reshape(n, 1)
    U = (M[i] + M[j]).reshape(n, 1)

    A = A + jnp.outer(J, J) - jnp.eye(n) * J
    M = M + (jnp.outer(U, U) - M) * U

    count += 1
    vals['count'] = count
    vals['A'] = A
    vals['M'] = M

    return vals, _


def pfkb(D : jnp.ndarray, ncc : int) -> tuple[jnp.ndarray, jnp.ndarray]:
    n = D.shape[0]
    nsteps = n - ncc
    A = prims_adjacency(D)
    mask = 1 - A + jnp.tril(jnp.ones((n, n))) # masks lower triangular, diagonal, and any edge not in tree.
    D_masked = D + (mask * D .max() * 2)
    smallest_k = argsort_first_k(D_masked.flatten(), nsteps)
    iinds, jinds = jnp.unravel_index(smallest_k, A.shape)

    vals = {
        'inds' : (iinds, jinds),
        'A' : jnp.eye(n),
        'M' : jnp.eye(n),
        'I' : jnp.eye(n),
        'count' : 0
    }
    vals, _  = jax.lax.scan(f=pfkb_step, init=vals, xs=None, length=nsteps)
    return vals['A'], vals['M']


