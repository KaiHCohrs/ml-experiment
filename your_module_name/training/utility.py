import torch
import numpy as np
import time
import random

from contextlib import contextmanager
from collections import OrderedDict, defaultdict


def copy_state(model):
    """
    Given PyTorch module `model`, makes a copy of the state onto CPU.
    Args:
        model: PyTorch module to copy state dict of

    Returns:
        A copy of state dict with all tensors allocated on the CPU
    """
    copy_dict = OrderedDict()
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        copy_dict[k] = v.cpu() if v.is_cuda else v.clone()

    return copy_dict

def early_stopping(
    model,
    objective,
    interval=5,
    patience=20,
    start=0,
    max_iter=1000,
    maximize=True,
    tolerance=1e-5,
    switch_mode=True,
    restore_best=True,
    tracker=None,
    scheduler=None,
    lr_decay_steps=1,
):

    """
    Early stopping iterator. Keeps track of the best model state during training. Resets the model to its
        best state, when either the number of maximum epochs or the patience [number of epochs without improvement)
        is reached.
    Also includes a convenient way to reduce the learning rate. Takes as an additional input a PyTorch scheduler object
        (e.g. torch.optim.lr_scheduler.ReduceLROnPlateau), which will automatically decrease the learning rate.
        If the patience counter is reached, the scheduler will decay the LR, and the model is set back to its best state.
        This loop will continue for n times in the variable lr_decay_steps. The patience and tolerance parameters in
        early stopping and the scheduler object should be identical.


    Args:
        model:     model that is being optimized
        objective: objective function that is used for early stopping. The function must accept single positional argument `model`
            and return a single scalar quantity.
        interval:  interval at which objective is evaluated to consider early stopping
        patience:  number of continuous epochs the objective could remain without improvement before the iterator terminates
        start:     start value for iteration (used to check against `max_iter`)
        max_iter:  maximum number of iterations before the iterator terminated
        maximize:  whether the objective is maximized of minimized
        tolerance: margin by which the new objective score must improve to be considered as an update in best score
        switch_mode: whether to switch model's train mode into eval prior to objective evaluation. If True (default),
                     the model is switched to eval mode before objective evaluation and restored to its previous mode
                     after the evaluation.
        restore_best: whether to restore the best scoring model state at the end of early stopping
        tracker (Tracker):
            Tracker to be invoked for every epoch. `log_objective` is invoked with the current value of `objective`. Note that `finalize`
            method is NOT invoked.
        scheduler:  scheduler object, which automatically reduces decreases the LR by a specified amount.
                    The scheduler's `step` method is invoked, passing in the current value of `objective`
        lr_decay_steps: Number of times the learning rate should be reduced before stopping the training.

    """
    training_status = model.training

    def _objective():
        if switch_mode:
            model.eval()
        ret = objective(model)
        if switch_mode:
            model.train(training_status)
        return ret

    def decay_lr(model, best_state_dict):
        old_objective = _objective()
        if restore_best:
            model.load_state_dict(best_state_dict)
            print("Restoring best model after lr decay! {:.8f} ---> {:.8f}".format(old_objective, _objective()))

    def finalize(model, best_state_dict):
        old_objective = _objective()
        if restore_best:
            model.load_state_dict(best_state_dict)
            print("Restoring best model! {:.8f} ---> {:.8f}".format(old_objective, _objective()))
        else:
            print("Final best model! objective {:.8f}".format(_objective()))

    epoch = start
    # turn into a sign
    maximize = -1 if maximize else 1
    best_objective = current_objective = _objective()
    best_state_dict = copy_state(model)

    for repeat in range(lr_decay_steps):
        patience_counter = 0

        while patience_counter < patience and epoch < max_iter:
            #print(current_objective)
            for _ in range(interval):
                epoch += 1
                if tracker is not None:
                    tracker.log_objective(current_objective)
                if (~np.isfinite(current_objective)).any():
                    print("Objective is not Finite. Stopping training")
                    finalize(model, best_state_dict)
                    return
                yield epoch, current_objective

            current_objective = _objective()

            # if a scheduler is defined, a .step with the current objective is all that is needed to reduce the LR
            if scheduler is not None:
                scheduler.step(current_objective)

            if current_objective * maximize < best_objective * maximize - tolerance:
                print(
                    "[{:03d}|{:02d}/{:02d}] ---> {}".format(epoch, patience_counter, patience, current_objective),
                    flush=True,
                )
                best_state_dict = copy_state(model)
                best_objective = current_objective
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    "[{:03d}|{:02d}/{:02d}] -/-> {}".format(epoch, patience_counter, patience, current_objective),
                    flush=True,
                )

        if (epoch < max_iter) & (lr_decay_steps > 1) & (repeat < lr_decay_steps):
            decay_lr(model, best_state_dict)

    finalize(model, best_state_dict)
    
@contextmanager
def eval_state(model):
    """
    Context manager, within which the model will be under `eval` mode.
    Upon existing, the model will return to whatever training state it
    was as it entered into the context.

    Args:
        model (PyTorch Module): PyTorch Module whose train/eval state is to be managed.

    Yields:
        PyTorch Module: The model switched to eval state.
    """
    training_status = model.training

    try:
        model.eval()
        yield model
    finally:
        model.train(training_status)
        

class Tracker:
    """
    Abstract Tracker class to serve as the bass class for all trackers.
    Defines the two interfacing methods of `log_objective` and `finalize`.

    """

    def log_objective(self, obj=None):
        """
        Logs the provided object

        Args:
            obj (Any, optional): Object to be logged

        Raises:
            NotImplementedError: Override this method to provide a functional behavior.
        """
        raise NotImplementedError("Please override this method to provide functional behavior")

    def finalize(self, obj):
        pass


class MultipleObjectiveTracker(Tracker):
    """
    Given a dictionary of functions, this will invoke all functions and log the returned values against
    the invocation timestamp. Calling `finalize` relativizes the timestamps to the first entry (i.e. first
    log_objective call) made.
    """

    def __init__(self, default_name=None, **objectives):
        """
        Initializes the tracker. Pass any additional objective functions as keywords arguments.

        Args:
            default_name (string, optional): Name under which the objective value passed into `log_objective` is saved under.
                If set to None, the passed in value is NOT saved. Defaults to None.
        """
        self._default_name = default_name
        self.objectives = objectives
        self.log = defaultdict(list)
        self.time = []

    def log_objective(self, obj=None):
        """
        Log the objective and also returned values of any list of objective functions this tracker
        is configured with. The passed in value of `obj` is logged only if `default_name` was set to
        something except for None.

        Args:
            obj (Any, optional): Value to be logged if `default_name` is not None. Defaults to None.
        """
        t = time.time()
        # iterate through and invoke each objective function and accumulate the
        # returns. This is performed separately from actual logging so that any
        # error in one of the objective evaluation will halt the whole timestamp
        # logging, and avoid leaving partial results
        values = {}
        # log the passed in value only if `default_name` was given
        if self._default_name is not None:
            values[self._default_name] = obj

        for name, objective in self.objectives.items():
            values[name] = objective()

        # Actually log all values
        self.time.append(t)
        for name, value in values.items():
            self.log[name].append(value)

    def finalize(self, reference=None):
        """
        When invoked, all logs are convereted into numpy arrays and are relativized against the first log entry time.
        Pass value `reference` to set the time relative to the passed-in value instead.

        Args:
            reference (float, optional): Timestamp to relativize all logged even times to. If None, relativizes all
                time entries to first time entry of the log. Defaults to None.
        """
        self.time = np.array(self.time)
        reference = self.time[0] if reference is None else reference
        self.time -= reference
        for k, l in self.log.items():
            self.log[k] = np.array(l)

    def asdict(self, time_key="time", make_copy=True):
        """
        Output the cotent of the tracker as a single dictionary. The time value


        Args:
            time_key (str, optional): Name of the key to save the time information as. Defaults to "time".
            make_copy (bool, optional): If True, the returned log times will be a (shallow) copy. Defaults to True.

        Returns:
            dict: Dictionary containing tracked values as well as the time
        """
        log_copy = {k: np.copy(v) if make_copy else v for k, v in self.log.items()}
        log_copy[time_key] = np.copy(self.time) if make_copy else self.time
        return log_copy

def set_random_seed(seed: int, deterministic: bool = True):
    """
    Set random generator seed for Python interpreter, NumPy and PyTorch. When setting the seed for PyTorch,
    if CUDA device is available, manual seed for CUDA will also be set. Finally, if `deterministic=True`,
    and CUDA device is available, PyTorch CUDNN backend will be configured to `benchmark=False` and `deterministic=True`
    to yield as deterministic result as possible. For more details, refer to
    PyTorch documentation on reproducibility: https://pytorch.org/docs/stable/notes/randomness.html

    Beware that the seed setting is a "best effort" towards deterministic run. However, as detailed in the above documentation,
    there are certain PyTorch CUDA opertaions that are inherently non-deterministic, and there is no simple way to control for them.
    Thus, it is best to assume that when CUDA is utilized, operation of the PyTorch module will not be deterministic and thus
    not completely reproducible.

    Args:
        seed (int): seed value to be set
        deterministic (bool, optional): If True, CUDNN backend (if available) is set to be deterministic. Defaults to True. Note that if set
            to False, the CUDNN properties remain untouched and it NOT explicitly set to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)  # this sets both CPU and CUDA seeds for PyTorch
