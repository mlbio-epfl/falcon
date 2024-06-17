import os
import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
from config import load_config, override_config

from loaders import construct_coarse2fine_loader
from models import create_backbone_factory
import faiss

def gather_representations(loader, model):
    num_samples = len(loader.dataset)
    actual = torch.zeros(num_samples, dtype=torch.long)
    actual_coarse = torch.zeros(num_samples, dtype=torch.long)
    feats = None
    bs = loader.batch_size
    with tqdm(total=len(loader.dataset)) as progress_bar:
        with torch.no_grad():
            for idx, data in enumerate(loader):
                x = data['inputs']
                y_fine = data['fine_label']
                y_coarse = data['coarse_label']
                x = x[1].cuda()
                y_fine = y_fine.cuda()
                actual[idx * bs: (idx + 1) * bs] = y_fine
                actual_coarse[idx * bs: (idx + 1) * bs] = y_coarse
                feats_ = model(x)
                if feats is None:
                    feats = torch.zeros(num_samples, feats_.shape[-1], dtype=torch.float32)
                feats[idx * bs: (idx + 1) * bs] = feats_.cpu()
                progress_bar.update(x.size(0))

    feats = F.normalize(feats, dim=-1)
    return feats, actual, actual_coarse

def neighours_with_coarse_(model, loader, num_neighbors, use_faiss, use_raw_input=False):
    model.eval() if not use_raw_input else None
    feats, actual, actual_coarse = gather_representations(loader, model)

    neighbors = torch.zeros((feats.shape[0], num_neighbors), dtype=torch.long)
    indices = torch.arange(feats.shape[0])
    for coarse_y in actual_coarse.unique():
        coarse_indices = indices[actual_coarse == coarse_y]
        if use_faiss:
            faiss_index = faiss.IndexFlatIP(feats.shape[-1])
            faiss_index.add(feats[actual_coarse == coarse_y].numpy())
            neighbors[actual_coarse == coarse_y] = coarse_indices[
                faiss_index.search(feats[actual_coarse == coarse_y].numpy(), num_neighbors + 1)[1][:, 1:]]
        else:
            _feat = feats[actual_coarse == coarse_y].cuda()
            _idx = (_feat @ _feat.T).topk(num_neighbors+1, dim=-1, largest=True)[1][:, 1:].cpu()
            neighbors[actual_coarse == coarse_y] = coarse_indices[_idx]
        print(f'Finished coarse class {coarse_y}')
    neighbors_classes = actual[neighbors]
    return neighbors, neighbors_classes, actual


def worker(device_id, cfg):
    nr=0
    cfg.RANK_ID = nr * cfg.SOLVER.DEVICES + device_id
    train_loader, val_loader, test_loader, fine_classes, coarse_classes = construct_coarse2fine_loader(cfg)

    train_loader = torch.utils.data.DataLoader(dataset=train_loader.dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                                               shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               pin_memory=cfg.DATALOADER.PIN_MEMORY)

    if not cfg.USE_RAW_INPUT:
        model, _ = create_backbone_factory(cfg)(cfg.MODEL.PRETRAINED)
    else:
        model = torch.nn.Identity()

    if hasattr(cfg.MODEL, 'WEIGHTS') and not cfg.USE_RAW_INPUT:
        new_state = {}
        for k,v in torch.load(cfg.MODEL.WEIGHTS)['model'].items():
            new_state[k.replace('module.', '')] = v
        model.load_state_dict(new_state, strict=False)
    model = model.cuda()

    num_neighbors = 20
    neighbors_with_coarse, neighbors_classes_with_coarse, actual = neighours_with_coarse_(model, train_loader, num_neighbors, cfg.USE_FAISS)
    correct_neighbors_with_coarse_percentage = (actual.unsqueeze(1).repeat(1, num_neighbors) == neighbors_classes_with_coarse).float().mean()
    print(f'Accuracy of neighbors w/ coarse: {correct_neighbors_with_coarse_percentage * 100}%')

    torch.save(neighbors_with_coarse, os.path.join(args.output_dir, f'{cfg.DATASET.NAME}_neighbors.pth'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--override_cfg', type=str, nargs='+', required=False)
    parser.add_argument('--port', type=str, required=False, default='8080')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--use_faiss', action='store_true')
    parser.add_argument('--use_raw_input', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.cfg_file)
    cfg = override_config(cfg, args.override_cfg) if args.override_cfg else cfg
    cfg.USE_FAISS = args.use_faiss
    cfg.USE_RAW_INPUT = args.use_raw_input
    os.makedirs(os.path.join(cfg.OUTPUT_DIR), exist_ok=True)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    cfg.SOLVER.DEVICES = 1
    if args.model:
        cfg.MODEL.WEIGHTS = args.model

    worker(0, cfg)
