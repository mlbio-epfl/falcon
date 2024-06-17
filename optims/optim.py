import torch.optim as optim

def construct_optim(optim_cfg, params):
    if optim_cfg.NAME == 'SGD':
        return optim.SGD(
            params,
            lr=optim_cfg.BASE_LR,
            momentum=optim_cfg.MOMENTUM,
            nesterov=optim_cfg.NESTEROV,
            weight_decay=optim_cfg.WEIGHT_DECAY
        )
    elif optim_cfg.NAME == 'Adam':
        return optim.Adam(
            params,
            lr=optim_cfg.BASE_LR,
            weight_decay=optim_cfg.WEIGHT_DECAY
        )
    elif optim_cfg.NAME == 'AdamW':
        return optim.AdamW(
            params,
            lr=optim_cfg.BASE_LR,
            weight_decay=optim_cfg.WEIGHT_DECAY
        )
    else:
        raise NotImplementedError('Optimizer not implemented: {}'.format(optim_cfg.NAME))


def construct_lr_scheduler(sched_cfg, optimizer):
    if not sched_cfg.NAME:
        return None
    elif sched_cfg.NAME == 'MultiStepLR':
        def get_idx(epoch):
            i = 0
            while i < len(sched_cfg.STEPS) and epoch >= sched_cfg.STEPS[i]:
                i += 1
            return i

        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: sched_cfg.GAMMA ** get_idx(epoch)
        )
    elif sched_cfg.NAME == 'StepLR':
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: sched_cfg.GAMMA ** (epoch // sched_cfg.STEP_SIZE)
        )
    elif sched_cfg.NAME == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.T_MAX,
            eta_min=sched_cfg.ETA_MIN
        )
    elif sched_cfg.NAME == 'CosineAnnealingWarmRestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.RESTART_PERIOD,
            eta_min=sched_cfg.ETA_MIN,
        )
    else:
        raise NotImplementedError('LR Scheduler not implemented: {}'.format(sched_cfg.NAME))