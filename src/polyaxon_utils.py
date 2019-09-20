import polyaxon_client
from polyaxon_client.tracking import (get_outputs_path, get_data_paths, Experiment)

import tensorflow as tf


def is_in_cluster():
    return polyaxon_client.settings.IN_CLUSTER


def is_managed():
    return polyaxon_client.settings.IS_MANAGED


def get_output_path(alternative):
    if not is_in_cluster():
        return alternative
    output_path = get_outputs_path()
    if output_path is None:
        output_path = alternative
    if not tf.io.gfile.exists(output_path):
        tf.io.gfile.makedirs(output_path)
    return output_path


def get_data_path(alternative):
    if not is_in_cluster():
        return alternative
    data_path = alternative
    data_paths = get_data_paths()
    if data_paths is not None and 'data' in data_paths:
        data_path = data_paths['data']
    return data_path


def get_experiment():
    return Experiment()


def set_params(params):
    if is_managed():
        experiment = get_experiment()
        experiment.log_params(**params)


class PolyaxonLoggingTensorHook(tf.estimator.LoggingTensorHook):
    """Hook that logs data to console and Polyaxon"""

    def __init__(self, tensors, experiment=None, every_n_iter=None, every_n_secs=None):
        super(PolyaxonLoggingTensorHook, self).__init__(tensors=tensors,
                                                        every_n_iter=every_n_iter,
                                                        every_n_secs=every_n_secs)
        self.experiment = experiment
        if is_managed():
            self.experiment = self.experiment or get_experiment()

    def _log_tensors(self, tensor_values):
        super(PolyaxonLoggingTensorHook, self)._log_tensors(tensor_values)

        if not self.experiment:
            return
        metrics = {k: tensor_values[k] for k in self._tensors.keys()}
        self.experiment.log_metrics(**metrics)
