import torch
from torch import nn
from tqdm import tqdm
from functools import partial

from .utility import early_stopping, MultipleObjectiveTracker, set_random_seed
from .losses import get_mseloss
from contextlib import contextmanager

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


def trainer_basic(model, dataloaders, seed,
            device='cuda', verbose=True, weight_decay = 0,
            interval=1, patience=10, epoch=0, lr_init=0.0005, max_iter=100, tolerance=1e-6, restore_best=True, betas=(0.9,0.999),
            lr_decay_steps=6, lr_decay_factor=0.3, min_lr=1e-8, track_training=False, disable_tqdm = False, **kwargs):
    
    
    def full_objective(model, inputs, target):
        return criterion(model(inputs.to(device)), target.to(device)) 

    ##### Model training #####################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()
    
    
    criterion = nn.MSELoss()
    stop_closure = partial(get_mseloss, dataloader = dataloaders['validation'], device=device)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': lr_init, 'betas': betas, 'weight_decay': weight_decay}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=lr_decay_factor, patience=patience,
                                                        threshold=tolerance,
                                                        min_lr=min_lr, verbose=verbose, threshold_mode='abs')
    n_iterations = len(dataloaders['train'])
    
    if track_training:
        tracker_dict = dict(mse_train=partial(get_mseloss, model=model, dataloader=dataloaders["train"], device=device),
                            mse_val=partial(get_mseloss, model=model, dataloader=dataloaders["validation"], device=device))
        if hasattr(model, 'tracked_values'):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None
        
    # train over epochs
    for epoch, val_obj in early_stopping(model, stop_closure, interval=interval, patience=patience,
                                        start=epoch, max_iter=max_iter, maximize=False,
                                        tolerance=tolerance, restore_best=restore_best, tracker=tracker,
                                        scheduler=scheduler, lr_decay_steps=lr_decay_steps):

        # print the quantities from tracker
        if verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        # train over batches
        optimizer.zero_grad()
        for batch_no, (inputs, target) in tqdm(enumerate(dataloaders["train"]), total=n_iterations,
                                            desc="Epoch {}".format(epoch), disable = disable_tqdm):

            loss = full_objective(model, inputs, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        
    ##### Model evaluation ####################################################################################################
    model.eval()
    tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    train_mse = get_mseloss(model, dataloaders["train"], device=device)
    validation_mse = get_mseloss(model, dataloaders["validation"], device=device)

    output=dict()
    output = {k: v for k, v in tracker.log.items()} if track_training else {}
    output["train_mse_loss"] = train_mse
    output["validation_mse_loss"] = validation_mse


    return output, model