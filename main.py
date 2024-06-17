import os
import argparse
import torch
from time import time_ns
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from config import load_config, override_config
from optims import construct_optim, construct_lr_scheduler
from loaders import construct_coarse2fine_loader
from models import create_backbone_factory, construct_classifier

from utils.logger import Logger
from utils.ema import ExponentialMovingAverage
from utils.utils import cosine_annealing, freeze_backbone_layers
from utils.graph_edit_distance import compute_ged, make_adjacency_matrix
from utils.discrete_optimization import compute_assignment_from_cost
from utils.snapshots import make_snapshot, load_snapshot

from train import train_one_epoch
from eval import eval


def worker(device_id, cfg):
    logger = Logger(filename=os.path.join(cfg.OUTPUT_DIR, 'log.txt')) if device_id == 0 else None
    logger.log(f"==== CONFIG ==== \n{cfg}\n =================") if device_id == 0 else None
    cudnn.benchmark = True

    nr = 0
    cfg.RANK_ID = nr * cfg.SOLVER.DEVICES + device_id
    cfg.DEVICE_ID = device_id
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=cfg.SOLVER.DEVICES,
        rank=cfg.RANK_ID
    )
    torch.cuda.set_device(device_id)

    cfg.NEIGHBORS = torch.load(cfg.NEIGHBORS) if cfg.NEIGHBORS is not None else None
    train_loader, val_loader, test_loader, fine_classes, coarse_classes = construct_coarse2fine_loader(cfg)
    print(fine_classes, coarse_classes)
    fine_classes = fine_classes if cfg.MODEL.NUM_CLASSES == None else cfg.MODEL.NUM_CLASSES

    model, embed_dim = create_backbone_factory(cfg)(cfg.MODEL.PRETRAINED)
    model = construct_classifier(cfg.MODEL.HEAD_TYPE, model, embed_dim, fine_classes).cuda(device_id)
    if cfg.MODEL.FROZEN:
        freeze_backbone_layers(model)
        logger.log('Backbone frozen') if device_id == 0 else None
    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
    model_params = model.parameters()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], find_unused_parameters=False) \
        if cfg.SOLVER.DEVICES > 1 else model
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) if cfg.SOLVER.DEVICES > 1 else model
    model_without_ddp = model.module if cfg.SOLVER.DEVICES > 1 else model

    if cfg.EVAL_ONLY and device_id == 0:
        logger = Logger(filename=os.path.join(cfg.OUTPUT_DIR, 'eval_log.txt'))
        state_dict = torch.load(cfg.MODEL.WEIGHTS)
        model.load_state_dict(state_dict['model'])
        M = state_dict['M']
        res = eval(model, val_loader)
        acc, ari = res['accuracy'], res['ari']
        logger.log(f'Val set ACC: {acc} ARI: {ari}')
        res = eval(model, test_loader)
        acc, ari = res['accuracy'], res['ari']
        logger.log(f'Test set ACC: {acc} ARI: {ari}')
        ged = compute_ged(make_adjacency_matrix(M.cpu().numpy()),
                          make_adjacency_matrix(test_loader.dataset.get_graph()))
        logger.log(f"GED: {ged}")
        return
    elif cfg.EVAL_ONLY:
        return

    optimizer = construct_optim(cfg.SOLVER.OPTIMIZER, model_params)
    scheduler = construct_lr_scheduler(cfg.SOLVER.LR_SCHEDULER, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.MODEL.PRECISION == 'fp16')

    M = compute_assignment_from_cost(
        torch.randn(fine_classes, coarse_classes).softmax(-1).numpy(),
        reg_coef=cfg.SOLVER.DISCRETE_OPTIM.BETA_REG, time_limit=cfg.SOLVER.DISCRETE_OPTIM.TIME_LIMIT).cuda()
    dist.broadcast(M, 0)
    initial_ged = compute_ged(
        make_adjacency_matrix(M.detach().cpu().numpy()),
        make_adjacency_matrix(test_loader.dataset.get_graph()))
    logger.log(f"Initial GED: {initial_ged}") if device_id == 0 else None

    if cfg.RESUME:
        start_epoch, model_without_ddp, M, ema, optimizer, scheduler = load_snapshot(
            cfg, model_without_ddp, ema, optimizer, scheduler, 'cuda')
    else:
        start_epoch = 1

    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS+1):
        tau = cosine_annealing(1, 0.,epoch-1, cfg.SOLVER.SOFT_LABELS_EPOCHS) \
            if epoch <= cfg.SOLVER.SOFT_LABELS_EPOCHS else 0.
        cfg.SOLVER.DISCRETE_OPTIM.BETA_REG = cfg.SOLVER.DISCRETE_OPTIM.BETA_REG if epoch <= cfg.SOLVER.SOFT_LABELS_EPOCHS else 0.
        logger.log(f'Tau: {tau}') if device_id == 0 else None

        t0 = time_ns()
        M = train_one_epoch(cfg, model, M, optimizer, train_loader, scaler, ema, tau=tau)
        t1 = time_ns()

        if scheduler is not None:
            scheduler.step()
            logger.log(f'Learning rate model: {scheduler.get_last_lr()}') if device_id == 0 else None

        if device_id == 0 and epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            model_time = (t1 - t0) / (10 ** 9)
            logger.log(f"Device {device_id} - Train time model: {model_time} sec ")
            res = eval(model_without_ddp, val_loader)
            acc, ari = res['accuracy'], res['ari']
            logger.log(f'Epoch {epoch} |Val set| ACC: {acc} ARI: {ari}')

            res = eval(model_without_ddp, test_loader)
            acc, ari = res['accuracy'], res['ari']
            logger.log(f'Epoch {epoch} |Test set| ACC: {acc} ARI: {ari}')
            ged = compute_ged(make_adjacency_matrix(M.detach().cpu().numpy()),
                        make_adjacency_matrix(test_loader.dataset.get_graph()))
            logger.log(f"GED: {ged}")

        if device_id == 0 and epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            torch.save({
                'model': model_without_ddp.state_dict(), 'M': M},
                os.path.join(cfg.OUTPUT_DIR, 'models', f"model_{epoch}.pth"))


        make_snapshot(cfg, epoch, model_without_ddp, M, ema, optimizer, scheduler) if device_id == 0 else None

    logger.close() if device_id == 0 else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--override_cfg', type=str, nargs='+', required=False)
    parser.add_argument('--port', type=str, required=False, default='8080')
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--resume', default=False, action='store_true')

    args = parser.parse_args()

    cfg = load_config(args.cfg_file)
    cfg = override_config(cfg, args.override_cfg) if args.override_cfg else cfg

    os.makedirs(os.path.join(cfg.OUTPUT_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, 'plots'), exist_ok=True)


    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    cfg.SOLVER.BATCH_SIZE //= cfg.SOLVER.DEVICES

    cfg.EVAL_ONLY = args.eval_only
    cfg.MODEL.WEIGHTS = args.model
    cfg.RESUME = args.resume

    if cfg.SOLVER.DEVICES > 1:
        mp.spawn(worker, nprocs=cfg.SOLVER.DEVICES, args=(cfg,))
    else:
        worker(0, cfg)
