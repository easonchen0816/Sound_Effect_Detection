import torch.optim as optim
import torch_optimizer as optim_new


def Adam(parameters, lr=0.0001, betas=(0.9, 0.999), weight_decay=0):
    """
    Args:
        parameters (iterable) – iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) – learning rate
        betas (Tuple[float, float], optional) – coefficients used for computing running averages of gradient and its square
        weight_decay (float, optional) – weight decay (L2 penalty)
    """
    return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay)


def AdamP(parameters, lr=0.0001, betas=(0.9, 0.999), weight_decay=0):
    """
    Args:
        parameters (iterable) – iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) – learning rate
        betas (Tuple[float, float], optional) – coefficients used for computing running averages of gradient and its square
        weight_decay (float, optional) – weight decay (L2 penalty)
    """
    return optim_new.AdamP(parameters, lr=lr, betas=betas, weight_decay=weight_decay)


def CosineAnnealingLR(optimizer, T_max):
    """
    Args:
        optimizer (Optimizer) – Wrapped optimizer.
        T_max (int) – Maximum number of iterations.
    """
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)


def ReduceLROnPlateau(optimizer, verbose=False):
    """Reduce learning rate when a metric has stopped improving.
       Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
       This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs, the learning rate is reduced.
    Args:
        optimizer (Optimizer) – Wrapped optimizer.
        verbose (bool) – If True, prints a message to stdout for each update.
    """
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=verbose)
