import numpy as np
from flax.training import train_state, orbax_utils
import ml_collections
import os, json, pickle
import tensorflow_datasets as tfds
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import orbax, optax
import jax
from tqdm import tqdm
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

'''
Contains classes for logging training / evaluation / testing
metrics (e.g. best seen)
'''

tf.config.set_visible_devices([], device_type='GPU')
np.random.seed(0)

class Metric():
    '''
    For storing values throughout training.
    '''
    def __init__(self, name : str, better : str = 'low'):
        self.name = name
        self.type='Metric'
        self.better = better
        self.vals = []
        self.steps = []
        self.best_val = None
        self.best_step = None

    def __iter__(self):
        yield "vals" , self.vals,
        yield "steps" , self.steps,
        yield "best_val" , self.best_val,
        yield "best_step" , self.best_step,
        yield "better" , self.better,

    def compare(self, new, old):
        if self.better == 'low':
            return new < old
        elif self.better == 'high':
            return new > old
        else:
            raise ValueError("better must be 'high' or 'low'")

    def update(self, step : int, val : float):
        self.vals.append(val)
        self.steps.append(step)


        if self.best_val is None:
            self.best_val = val
            self.best_step = step
            has_improved = True
        else:
            if self.compare(val, self.best_val):
                self.best_val = val
                self.best_step = step
                has_improved = True
            else:
                has_improved = False
        return has_improved


class AvgMetric():
    '''
    For storing values throughout training and averaging at moments.
    '''
    def __init__(self, name : str, better : str = 'low'):
        self.name = name
        self.type = 'AvgMetric'
        self.better = better

        self.vals = []
        self.best_val = None

        self.steps = []
        self.best_step = None

        self.avg_vals = []
        self.best_avg_val = None

        self.avg_steps = []
        self.best_avg_step = None

        self.curr_count = 0

    def __iter__(self):
        yield "vals" , self.vals,
        yield "steps" , self.steps,
        yield "best_val" , self.best_val,
        yield "best_step" , self.best_step,
        yield "better" , self.better,
        yield "avg_vals", self.avg_vals,
        yield "best_avg_val" , self.best_avg_val,
        yield "best_avg_step" , self.best_avg_step,
        yield "avg_steps", self.avg_steps,
        yield "curr_count", self.curr_count


    def compare(self, new, old):
        if self.better == 'low':
            return new < old
        elif self.better == 'high':
            return new > old
        else:
            raise ValueError("better must be 'high' or 'low'")

    def store(self, step : int, val : float):
        self.vals.append(val)
        self.steps.append(step)
        self.curr_count += 1

        if self.best_val is None:
            self.best_val = val
            self.best_step = step
            has_improved = True
        else:
            if self.compare(val, self.best_val):
                self.best_val = val
                self.best_step = step
                has_improved = True
            else:
                has_improved = False
        return has_improved


    def update(self, step : int):
        # store the count and reset
        count = self.curr_count
        self.curr_count = 0

        avg_val = np.mean(self.vals[-count:])

        self.avg_steps.append(step)
        self.avg_vals.append(avg_val)

        if self.best_avg_val is None:
            self.best_avg_val = avg_val
            self.best_avg_step = step
            has_improved = True
        else:
            if self.compare(avg_val, self.best_avg_val):
                self.best_avg_val = avg_val
                self.best_avg_step = step
                has_improved = True
            else:
                has_improved = False
        return has_improved


class MetricDict():
    def __init__(self, metric_args : dict):
        self.metrics = {}
        self.active = []
        for k, v in metric_args.items():
            self.metrics[k] = self.newmetric(k=k, kdict=v)


    def newmetric(self, k, kdict):
        kwargs = {k : v for (k, v) in kdict.items() if k!='type'}
        if kdict['type'] == 'AvgMetric':
            return AvgMetric(name=k, **kwargs)
        elif kdict['type'] == 'Metric':
            return Metric(name=k, **kwargs)
        else:
            raise ValueError("Type of Metric is Undefined")

    def __iter__(self):
        for k, v in self.metrics.items():
            yield k, dict(v)
        yield 'active' , self.active

    def store(self, step: int, updates : dict):
        improvements = {}
        for k, v in updates.items():
            if k not in self.active:
                self.active.append(k)
            improvements[k] = self.metrics[k].store(step=step, val=v)
        return improvements

    def update(self, step : int, updates : dict):
        improvements = {}
        for k, v in updates.items():
            improvements[k] = self.metrics[k].update(step=step, val=v)

        for k in self.active:
            improvements[k] = self.metrics[k].update(step=step)
        self.active = []
        return improvements

    def get_prev(self):
        prev_vals = {}
        for k, v in self.metrics.items():
            if v.type == 'AvgMetric':
                try:
                    prev_vals[k] = v.avg_vals[-1]
                except IndexError:
                    prev_vals[k] = None
            elif v.type == 'Metric':
                try:
                    prev_vals[k] = v.vals[-1]
                except IndexError:
                    prev_vals[k] = None

            else:
                raise ValueError
        return prev_vals

    def print_prev(self, step: int, prev_vals : dict):
        summary = f"step {step} : "
        for k, v in prev_vals.items():
            summary += str(k) + f" = {str(v)},  "
        print(summary)


    def write_scalar_values(self, step, writer, save=False):
        summary = self.get_prev()
        summary = {k : v for (k, v) in summary.items() if v is not None}
        if save:
            writer.write_scalars(step, summary)
        else:
            self.print_prev(step, summary)


def fold_in_dict(d, i):
    return { k : jax.random.fold_in(v, i) for (k, v) in d.items()}

def fold_in_key(d, i, key):
    d[key] = jax.random.fold_in(d[key], i)
    return d

