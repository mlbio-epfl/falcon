from tqdm import tqdm
import torch
from utils.metrics import adjusted_rand_score, cluster_acc, compute_matchings, macro_accuracy

def eval(model, val_loader):
    model.eval()

    actual = []
    pred = []

    total_conf = 0.

    with tqdm(total=len(val_loader.dataset)) as progress_bar:
        with torch.no_grad():
            for data in val_loader:
                x = data['inputs']
                y_fine = data['fine_label']
                y_coarse = data['coarse_label']

                x = x.cuda()
                y_fine = y_fine.cuda()

                actual.append(y_fine)

                logits = model(x)
                conf, pred_ = logits.softmax(-1).max(-1)
                total_conf += conf.sum().item()

                pred.append(pred_)
                progress_bar.update(x.size(0))

    actual = torch.cat(actual, dim=0).cpu()
    _pred = torch.cat(pred, dim=0).cpu()

    ars = adjusted_rand_score(_pred, actual)
    avg_conf = total_conf / len(val_loader.dataset)
    acc = cluster_acc(_pred, actual)
    mapper = compute_matchings(_pred, actual)
    macc = macro_accuracy(mapper[_pred], actual)

    return {
        'accuracy': acc,
        'ari': ars,
        'avg_conf': avg_conf,
        'macc': macc
    }

