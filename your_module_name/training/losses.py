from torch import nn

from .utility import eval_state

def get_mseloss(model, dataloader, device='cpu', **kwargs):

    mseloss = 0
    loss = nn.MSELoss(reduction = 'sum')
    
    with eval_state(model):
        for inputs, target in dataloader:
            outputs = model(inputs.to(device))
            mseloss += loss(target.to(device), outputs)

    return (mseloss/len(dataloader.dataset)).detach().cpu().numpy()
