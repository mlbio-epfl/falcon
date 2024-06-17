import torch
import os

def make_snapshot(cfg, epoch, model_without_ddp, M, ema, optimizer, scheduler):
    torch.save({
        'epoch': epoch,
        'model': model_without_ddp.state_dict(),
        'M': M,
        'ema': ema.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, os.path.join(cfg.OUTPUT_DIR, 'snapshot.pth'))


def load_snapshot(cfg, model_without_ddp, ema, optimizer, scheduler, device):
    path = os.path.join(cfg.OUTPUT_DIR, 'snapshot.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f'No snapshot found at {cfg.OUTPUT_DIR / f"snapshot.pth"}')

    checkpoint = torch.load(path, map_location=device)

    model_without_ddp.load_state_dict(checkpoint['model'])
    M = checkpoint['M']
    ema.load_state_dict(checkpoint['ema'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch, model_without_ddp, M, ema, optimizer, scheduler
