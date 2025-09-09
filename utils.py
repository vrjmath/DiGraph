import torch
from sklearn.metrics import f1_score, roc_auc_score

def compute_metrics(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # Cast node features to float
            data.x = data.x.float()
            out = torch.sigmoid(model(data.x, data.edge_index, data.batch)).squeeze()
            ys.append(data.y.cpu())
            preds.append(out.cpu())
    ys = torch.cat(ys)
    preds = torch.cat(preds)
    
    # compute F1, ROC-AUC
    f1 = f1_score(ys, preds.round())
    roc_auc = roc_auc_score(ys, preds)
    return {"f1": f1, "roc_auc": roc_auc}

