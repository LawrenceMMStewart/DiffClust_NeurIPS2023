import jax
import jax.numpy as jnp
import numpy as np
from forests import make_perturbed_mwst, make_perturbed_mwst_constrained, pairwise_square_distance, mwst, mwst_constrained


np.random.seed(0)

def test_gradFk_is_Ak_unconstrained():
    """
    test that gradient of F_sigma
    is each to the argmax A_sigma
    for the unconstrained perturbed LP
    """

    for i in range(3):
        pmwst = jax.jit(make_perturbed_mwst())

        X = np.random.randn(32, 2)
        S = - pairwise_square_distance(X)
        key = jax.random.PRNGKey(0)

        for ncc in [5, 10, 20]:

            def pF(S):
                A, F, M = pmwst(S, ncc, key)
                return F

            g = jax.grad(pF)(S)
            A, _, _ = pmwst(S, ncc, key)

            assert jnp.isclose(g, A).all()



def test_gradFk_isAk_constrained_groundtruthM():
    """
    test that gradient of F_sigma(., M)
    is each to the argmax A_sigma(., M)
    for the constrained perturbed LP,
    where M true value of the unconstrained unperturbed LP
    (i.e. F_sigma(., M) A_sigma(., M) are
    effectively unconstrained).
    """
    pmwst_constrained = jax.jit(make_perturbed_mwst_constrained())
    jmwst = jax.jit(mwst)

    for i in range(3):

        X = np.random.randn(32, 2)
        S = - pairwise_square_distance(X)
        key = jax.random.PRNGKey(0)

        for ncc in [5, 10, 20]:
            Atrue, Mtrue = jmwst(S, ncc)
            Ctrue = 2 * Mtrue - 1
            Amtrue, Mmtrue = mwst_constrained(S, Ctrue, ncc)
            assert jnp.isclose(Atrue, Amtrue).all()
            assert jnp.isclose(Mtrue, Mmtrue).all()

            def cpF(S):
                A, F, M = pmwst_constrained(S, Ctrue, ncc, key)
                return F

            A, F, M = pmwst_constrained(S, Ctrue, ncc, key)
            g = jax.grad(cpF)(S)

            assert jnp.isclose(g, A).all()




def test_gradFk_isAk_constrained_unconstrainedM():
    """
    Same as test_gradFk_isAk_constrained_groundtruthM
    but M is the matrix of 1/2s i.e. unconstrained
    """
    pmwst_constrained = jax.jit(make_perturbed_mwst_constrained())
    for i in range(3):
        X = np.random.randn(32, 2)
        S = - pairwise_square_distance(X)
        key = jax.random.PRNGKey(0)

        for ncc in [5, 10, 20]:

            Mtarget = jnp.ones_like(S) * 0.5
            C_star = jnp.zeros_like(S) # 2Mtarget-  1

            def cpF(S):
                A, F, M = pmwst_constrained(S, C_star, ncc, key)
                return F


            A, F, M = pmwst_constrained(S, C_star, ncc, key)
            g = jax.grad(cpF)(S)
            assert jnp.isclose(g, A).all()



def test_gradFk_isAk_constrained_Msimple():
    """
    test that gradient of F_sigma(., M)
    is each to the argmax A_sigma(., M)
    for the constrained perturbed LP,
    where M is unconstrained apart from one
    "must-link" and one "must not-link" constraint.
    """

    pmwst_constrained = jax.jit(make_perturbed_mwst_constrained())
    for i in range(3):
        X = np.random.randn(32, 2)
        S = - pairwise_square_distance(X)
        key = jax.random.PRNGKey(0)

        for ncc in [5, 10, 20]:

            Mtarget = np.ones_like(S) * 0.5
            Mtarget[1, 2] = 0
            Mtarget[2, 1] = 0

            Mtarget[5, 4] = 1
            Mtarget[4, 5] = 1

            C_star = 2 * Mtarget - 1

            def cpF(S):
                A, F, M = pmwst_constrained(S, C_star, ncc, key)
                return F

            A, F, M = pmwst_constrained(S, C_star, ncc, key)
            g = jax.grad(cpF)(S)

            assert jnp.isclose(g, A).all()



def test_gradMk_is_zero_unconstrained():
    """
    tests that the perturbed LP
    returns a jvp of zero w.r.t M
    """

    pmwst = jax.jit(make_perturbed_mwst())
    for i in range(3):

        X = np.random.randn(32, 2)
        S = - pairwise_square_distance(X)
        key = jax.random.PRNGKey(0)

        for ncc in [5, 10, 20]:
            def f(S):
                A, F, M = pmwst(S, ncc, key)
                return M.sum()

            g = jax.grad(f)(S)
            assert jnp.isclose(g, jnp.zeros_like(g)).all()


def test_gradMk_is_zero_constrained():
    """
    tests that the perturbed constrained LP
    returns a jvp of zero w.r.t M
    """


    pmwst_constrained = jax.jit(make_perturbed_mwst_constrained())
    for i in range(3):

        X = np.random.randn(32, 2)
        S = - pairwise_square_distance(X)
        key = jax.random.PRNGKey(0)

        for ncc in [5, 10, 20]:
            def f(S):
                A, F, M = pmwst_constrained(S, jnp.zeros_like(S), ncc, key)
                return M.sum()

            g = jax.grad(f)(S)

            assert jnp.isclose(g, jnp.zeros_like(g)).all()


