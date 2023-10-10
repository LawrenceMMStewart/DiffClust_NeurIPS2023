import numpy as np
import pickle, sys, os, json, glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import jax
import jax.numpy as jnp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state



def load_train_embeddings(load_dir='.'):
    file = glob.glob(os.path.join(load_dir, 'embedding-downstream*'))[0]
    with open(file, 'rb') as f:
        X, Y = pickle.load(f)
    return X, Y

def load_test_embeddings(load_dir='.'):
    file = glob.glob(os.path.join(load_dir, 'embedding-test*'))[0]
    with open(file, 'rb') as f:
        X, Y = pickle.load(f)
    return X, Y


def fit_linear(load_dir, N_downstream=-1):

    X_train, Y_train = load_train_embeddings(load_dir)
    X_test, Y_test = load_test_embeddings(load_dir)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    if N_downstream == -1:
        N_downstream = 1000

    X_train = X_train[:N_downstream]
    Y_train = Y_train[:N_downstream]

    clf = LogisticRegression(solver='saga', max_iter=50000)
    print("Fitting model")
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)

    err = 1 - score
    print(f"Linear head on {load_dir}, achieved err = {err}")
    return err


def main(dir_of_dirs, save=False, N_downstream=-1):
    dirs = glob.glob(f'{dir_of_dirs}/*')
    for d in dirs:
        if len(glob.glob(os.path.join(d, 'embedding-test*'))) > 0:
            if len(glob.glob(os.path.join(d, 'linearprobe.txt'))) == 0:
                err = fit_linear(d, N_downstream)
                if save:
                    with open(os.path.join(d, 'linearprobe.txt'), 'w') as f:
                        f.write(str(err.item()))




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='Downstream Transfer Learning',
                    description='Train a head on top of a frozen model on downstream dataset',
                    epilog='Transfer Learning experiment.')


    parser.add_argument('--N_downstream', default=-1, type=int, help='number of downstream data points (max = 1000 = -1)')

    parser.add_argument('--path', default='my_exps', type=str, help='dir of dirs')

    parser.add_argument('--save', action='store_true', help='save test result on linear probe')

    args = parser.parse_args()
    args = vars(args)

    main(args['path'], save=args['save'], N_downstream=args['N_downstream'])





