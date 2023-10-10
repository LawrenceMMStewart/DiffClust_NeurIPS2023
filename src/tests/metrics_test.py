from utils import MetricDict
import numpy as np



def test_metric_logging():

    metric_args = {
        'pert_pl_train' : {'type' : 'AvgMetric', 'better' : 'low'},
        'pert_pl_eval' : {'type' : 'Metric', 'better' : 'low'},
        'acc_eval' : {'type' : 'Metric', 'better' : 'high'},
    }

    metrics = MetricDict(metric_args)


    # first log

    imp = metrics.update(step=0, updates={'pert_pl_eval' : 1.7, 'acc_eval' : 0.1})

    assert metrics.metrics['pert_pl_eval'].vals == [1.7]
    assert metrics.metrics['pert_pl_eval'].steps == [0]
    assert metrics.metrics['pert_pl_eval'].best_val == 1.7
    assert metrics.metrics['pert_pl_eval'].best_step == 0

    assert metrics.metrics['acc_eval'].vals == [0.1]
    assert metrics.metrics['acc_eval'].steps == [0]
    assert metrics.metrics['acc_eval'].best_val == 0.1
    assert metrics.metrics['acc_eval'].best_step == 0

    assert imp == {'pert_pl_eval' : True, 'acc_eval' : True}


    # second log

    imp = metrics.update(step=1, updates={'pert_pl_eval' : 2.3, 'acc_eval' : 0.2})

    assert metrics.metrics['pert_pl_eval'].vals == [1.7, 2.3]
    assert metrics.metrics['pert_pl_eval'].steps == [0, 1]
    assert metrics.metrics['pert_pl_eval'].best_val == 1.7
    assert metrics.metrics['pert_pl_eval'].best_step == 0

    assert metrics.metrics['acc_eval'].vals == [0.1, 0.2]
    assert metrics.metrics['acc_eval'].steps == [0, 1]
    assert metrics.metrics['acc_eval'].best_val == 0.2
    assert metrics.metrics['acc_eval'].best_step == 1

    assert imp == {'pert_pl_eval' : False, 'acc_eval' : True}


    # some average logging


    imp = metrics.store(step=3, updates={'pert_pl_train' : 2})
    assert imp == {'pert_pl_train' : True}
    imp = metrics.store(step=4, updates={'pert_pl_train' : 3})
    assert imp == {'pert_pl_train' : False}
    imp = metrics.store(step=5, updates={'pert_pl_train' : 1})
    assert imp == {'pert_pl_train' : True}

    assert metrics.metrics['pert_pl_train'].vals == [2, 3, 1]
    assert metrics.metrics['pert_pl_train'].steps == [3, 4, 5]
    assert metrics.metrics['pert_pl_train'].best_val == 1.0
    assert metrics.metrics['pert_pl_train'].best_step == 5
    assert metrics.metrics['pert_pl_train'].curr_count == 3

    assert metrics.active == ['pert_pl_train']


    # third log, leading to averaging of average logging

    imp = metrics.update(step=6, updates={'pert_pl_eval' : 0.3, 'acc_eval' : 0.2})

    assert metrics.metrics['pert_pl_train'].avg_vals == [2.]
    assert metrics.metrics['pert_pl_train'].avg_steps == [6]
    assert metrics.metrics['pert_pl_train'].best_avg_val == 2.
    assert metrics.metrics['pert_pl_train'].best_avg_step == 6
    assert metrics.metrics['pert_pl_train'].vals == [2, 3, 1]
    assert metrics.metrics['pert_pl_train'].steps == [3, 4, 5]
    assert metrics.metrics['pert_pl_train'].best_val == 1.0
    assert metrics.metrics['pert_pl_train'].best_step == 5
    assert metrics.metrics['pert_pl_train'].curr_count == 0

    assert metrics.metrics['pert_pl_eval'].vals == [1.7, 2.3, 0.3]
    assert metrics.metrics['pert_pl_eval'].steps == [0, 1, 6]
    assert metrics.metrics['pert_pl_eval'].best_val == 0.3
    assert metrics.metrics['pert_pl_eval'].best_step == 6

    assert metrics.metrics['acc_eval'].vals == [0.1, 0.2, 0.2]
    assert metrics.metrics['acc_eval'].steps == [0, 1, 6]
    assert metrics.metrics['acc_eval'].best_val == 0.2
    assert metrics.metrics['acc_eval'].best_step == 1


    #  some more average logs before final log
    imp = metrics.store(step=7, updates={'pert_pl_train' : 0.5})
    assert imp == {'pert_pl_train' : True}
    imp = metrics.store(step=8, updates={'pert_pl_train' : 0.5})
    assert imp == {'pert_pl_train' : False}
    imp = metrics.store(step=9, updates={'pert_pl_train' : 0.5})
    assert imp == {'pert_pl_train' : False}
    imp = metrics.store(step=10, updates={'pert_pl_train' : 0.5})
    assert imp == {'pert_pl_train' : False}

    assert metrics.metrics['pert_pl_train'].vals == [2, 3, 1] + [0.5] * 4
    assert metrics.metrics['pert_pl_train'].steps == [3, 4, 5, 7, 8, 9, 10]
    assert metrics.metrics['pert_pl_train'].best_val == 0.5
    assert metrics.metrics['pert_pl_train'].best_step == 7
    assert metrics.metrics['pert_pl_train'].curr_count == 4

    assert metrics.active == ['pert_pl_train']


    # final log (no eval)
    im = metrics.update(step=11, updates = {})
    assert im == {'pert_pl_train' : True}
    assert metrics.active == []
    assert metrics.metrics['pert_pl_train'].avg_vals == [2, 0.5]
    assert metrics.metrics['pert_pl_train'].avg_steps == [6, 11]
    assert metrics.metrics['pert_pl_train'].best_avg_step == 11
    assert metrics.metrics['pert_pl_train'].best_avg_val == 0.5


    prev = metrics.get_prev()
    assert prev == {'pert_pl_train' : 0.5, 'pert_pl_eval' : 0.3, 'acc_eval' : 0.2}


