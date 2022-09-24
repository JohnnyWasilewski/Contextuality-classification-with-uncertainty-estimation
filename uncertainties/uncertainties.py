import torch
import torch.nn as nn

import numpy as np
import laplace

def entropy(x):
    return -1 * (x*np.log2(x + 0.000000000001) + (1-x)*np.log2(1-x + 0.000000000001))

# ------------------------------------------------------------
# ------------------------- LAPLACE --------------------------
# ------------------------------------------------------------

def get_preds_laplace(train_dataloader, predict_dataloader, model, device):
    model.to('cpu')
    la = laplace.Laplace(model, likelihood='classification')
    la.fit(train_dataloader)
    preds = list()
    for X, _ in predict_dataloader:
        preds.append(la.predictive_samples(x=X, n_samples=1000)[:, :, 1])
    m = torch.concat(preds, dim=1).numpy()
    return m # np.sqrt(np.var(m, axis=0)), np.mean(m, axis=0)

# -------------------------------------------------------------------------
# ------------------------------ DROPOUT ----------------------------------
# --------------------------------------------------------------------------
def get_preds_dropout(dataloader, model, device):
    model.eval()
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

    iter_num = 100
    y_true, y_preds, y_preds_raw = [], [], []
    softmax = nn.Softmax(dim=1)
    k,l=[],[]
    for X, y in dataloader:
        y_preds_tmp, y_preds_raw_tmp = [], []
        for _ in range(iter_num):
            X = X.to(device)
            y_hat = softmax(model(X))[:, 1].cpu().detach().numpy()
            l.append(softmax(model(X))[:, 0].cpu().detach().numpy())
            y_preds_tmp.append(y_hat)
            y_preds_raw_tmp.append(model(X)[:, 1].cpu().detach().numpy())
        y_true.append(y.numpy())
        y_preds.append(np.array(y_preds_tmp))
        y_preds_raw.append(np.array(y_preds_raw_tmp))
        k.append(np.array(l))
    raw = np.concatenate(y_preds_raw, axis=1)
    m = np.concatenate(y_preds, axis=1)
    unc = entropy(np.count_nonzero(m < 0.5, axis=0) / iter_num)
    unc_raw = entropy(np.count_nonzero(raw < 0.5, axis=0) / iter_num)
    return m


def calculate_uncertainties(arr: np.array):
    T = arr.shape[0]
    pred_sum = np.sum(arr, axis=0)
    entropy_uncertainty = entropy(pred_sum/T)
    mutual_info = entropy_uncertainty + 1/T * np.sum(entropy(arr), axis=0)
    var_ratio = 1 - np.count_nonzero(arr < 0.5, axis=0)/T

    aleatoric_uncertainty = np.mean(entropy(arr), axis=0)
    return {"entropy": entropy_uncertainty,
            "mutual_info": mutual_info,
            "var_ratio": var_ratio,
            "aleatoric_uncertainty": aleatoric_uncertainty
            }, [entropy_uncertainty, mutual_info, var_ratio, aleatoric_uncertainty]
