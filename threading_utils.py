from typing import *

import numpy as np

import threading

class MultiThreadWorker(object):

    DEBUG_MODE = True

    def __init__(self, num_workers : int, target_function : Callable) -> None:

        if (self.DEBUG_MODE):

            assert isinstance(num_workers, int)

            assert callable(target_function)

        self._num_workers = num_workers

        self._target_function = target_function

    def process(self, data : np.ndarray, *args, divide_axis : Optional[ int ] = 0, concat_axis : Optional[ int ] = None, **kwargs) -> np.ndarray:

        if (self.DEBUG_MODE):

            assert isinstance(data, np.ndarray)

            assert isinstance(divide_axis, int)

            assert ((concat_axis is None) or (isinstance(concat_axis, int)))

        if (concat_axis is None):
            concat_axis = divide_axis

        workload = int(np.ceil(data.shape[divide_axis] / self._num_workers))

        data_batches = []

        indexes = [ slice(None, None) ] * len(data.shape)

        for start_idx in range(0, data.shape[divide_axis], workload):

            end_idx = min(start_idx + workload, data.shape[divide_axis])

            data_batches.append(data[(*indexes[:divide_axis], slice(start_idx, end_idx), *indexes[divide_axis+1:])])

        num_workers = min(self._num_workers, len(data_batches))

        threads = []

        processed_data = [ None ] * num_workers

        def process_data(worker_id : int, *args, **kwargs) -> None:

            nonlocal processed_data, data_batches

            processed_data[worker_id] = self._target_function(data_batches[worker_id], *args, **kwargs)

        for worker_id in range(num_workers):

            thread = threading.Thread(target = process_data, args = [ worker_id, *args ], kwargs = {**kwargs})

            thread.start()

            threads.append(thread)

        for thread in reversed(threads):
            thread.join()

        return np.concatenate(processed_data, axis = concat_axis)