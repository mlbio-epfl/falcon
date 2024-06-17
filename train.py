import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from utils.distributed_utils import gather_from_all
from utils.discrete_optimization import compute_assignment_from_cost
from utils.utils import entropy

def train_one_epoch(cfg, model, M, optimizer, train_loader, scaler, ema, tau):
    model.train()

    num_fine, num_coarse = M.shape

    cls_loss_coarse = 0
    cls_loss_fine = 0
    reg_loss = 0
    total_loss = 0
    iters = 0

    actual_coarse_all = []
    preds_all = []

    for i, data in enumerate(train_loader):
        x = data['inputs']
        _y_fine = data['fine_label']
        y_coarse = data['coarse_label']
        neighbors = data['neighbors']

        N, num_neig = neighbors.shape[:2]
        pattern = 'n k c h w -> (n k) c h w' if len(neighbors.shape) == 5 else 'n k c -> (n k) c'
        neighbors = rearrange(neighbors, pattern)
        x, x_ema = x[0].cuda(), x[1].cuda()
        actual_coarse_all.append(y_coarse)
        y_coarse = y_coarse.cuda()

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=cfg.MODEL.PRECISION == 'fp16'):
            with torch.no_grad():
                with ema.average_parameters():
                    logits_ema = model(x_ema).detach()
                    ema_probs = logits_ema.softmax(-1)
                    preds_all.append(ema_probs)

                    neighbors = rearrange(model(neighbors.cuda()).softmax(-1).detach(),
                                          '(n k) c -> n k c', n=N, k=num_neig)

            logits = model(x)
            probs = logits.softmax(-1)

            log_coarse_prob = torch.logsumexp(
                logits.unsqueeze(-1).repeat(1,1, M.shape[1]).masked_fill_(~M.unsqueeze(0).repeat(logits.shape[0], 1, 1).bool(), float('-inf')), dim=1
            ) - torch.logsumexp(logits, dim=-1, keepdim=True)

            loss_consist = - torch.einsum('n c, n k c -> n k', probs, neighbors).log().mean()
            assert loss_consist.requires_grad

            mask = ~M.T[y_coarse].bool()
            q_fine_soft = (logits_ema / cfg.SOLVER.LOSS.TEMP).masked_fill(mask, float('-inf')).softmax(-1)

            q_fine_hard = F.one_hot(q_fine_soft.argmax(-1), q_fine_soft.shape[-1]).float()
            pseudo_fine_lbls = tau * q_fine_soft + (1 - tau) * q_fine_hard

            loss_cls_fine = - (F.log_softmax(logits, dim=-1) * pseudo_fine_lbls).sum(-1).mean()
            loss_cls_coarse = F.cross_entropy(log_coarse_prob, y_coarse)

            avg_prob = gather_from_all(probs).mean(0)
            loss_reg_fine = - entropy(avg_prob) + math.log(probs.shape[1])

            loss_total = cfg.SOLVER.LOSS.LAMBDA_1 * loss_cls_coarse \
                        + cfg.SOLVER.LOSS.LAMBDA_2 * (loss_cls_fine + loss_consist) \
                        + cfg.SOLVER.LOSS.LAMBDA_3 * loss_reg_fine

        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update()

        cls_loss_fine += loss_cls_fine.item()
        cls_loss_coarse += loss_cls_coarse.item()
        reg_loss += loss_reg_fine.item()
        total_loss += loss_total.item()
        iters += 1

        if iters % 10 == 0:
            print(f"Total loss: {total_loss/iters:.4f} Cls Fine: {cls_loss_fine / iters:.4f}, Cls Coarse: {cls_loss_coarse / iters:.4f}, Reg: {reg_loss / iters:.4f}")

        if i % cfg.SOLVER.DISCRETE_OPTIM.SOLVE_EVERY == 0 and i > 0:
            fine_preds =  gather_from_all(torch.cat(preds_all, dim=0).cuda())
            coarse_gt = gather_from_all(torch.cat(actual_coarse_all, dim=0).cuda())
            coarse_gt_oh = F.one_hot(coarse_gt, num_coarse).float()
            cost = (fine_preds.T @ coarse_gt_oh) / coarse_gt_oh.shape[0]
            M = compute_assignment_from_cost(cost.cpu().numpy(), reg_coef=cfg.SOLVER.DISCRETE_OPTIM.BETA_REG, time_limit=cfg.SOLVER.DISCRETE_OPTIM.TIME_LIMIT).cuda()
            dist.broadcast(M, 0)
            preds_all, actual_coarse_all = [], []

    return M
