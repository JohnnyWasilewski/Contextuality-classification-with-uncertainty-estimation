from pathlib import Path
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

import uncertainties as unc
import dataloaders as dl
import models as m


def experiment(num, idx, device, metrics, results_path, batch_size=20, save_fig=True, test_num=500):
    data_loader_train, data_loader_test, data_loader_train_r, data_loader_test_r = dl.get_dataloaders(num, batch_size)
    contextual_data_loader = dl.get_dataloaders_c(test_num, batch_size)
    print('jestem!')
    classifier, classifier_raw, acc, acc_raw = m.learn_classifier(data_loader_train, data_loader_test, device)
    laplace = unc.get_preds_laplace(contextual_data_loader, classifier_raw, classifier, data_loader_train, 'cpu')
    dropout = unc.get_preds_dropout(contextual_data_loader, classifier_raw, 'cpu')

    print('policzyłem niepewności!')

    dists = list()
    for X, y in contextual_data_loader:
        dists.extend([dl.projection_distance(x.numpy()) for x in X])
    metrics[(num, idx, "acc", "raw")] = acc_raw
    metrics[(num, idx, "acc_adjusted", "raw")] = acc
    print('Policzyłem dystanse!')
    print_corr_results(pearsonr, dists, [*laplace[:3], dropout], num, idx, metrics, save_fig, results_path)
    return dropout, laplace, dists, acc, acc_raw


def print_corr_results(correlation_metric, preds_regressor, preds_uncertainty, num, idx, corr, save_fig, folder_name='NL_square3'):
    for preds, name in zip(preds_uncertainty, ["raw", "apx", "MCMC", "MCDO"]):
        preds_adjusted = list(map(lambda x: 4*(x-0.5)**2, preds))
        corr[(num, idx, name, "raw")] = correlation_metric(preds_regressor, preds)[0]
        corr[(num, idx, name, "adjusted")] = correlation_metric(preds_regressor, preds_adjusted)[0]

    fig, ax = plt.subplots(4, 1, clear=True, figsize=(7, 7))
    if idx == 0:
        Path(f'{folder_name}/experiment_{num}').mkdir(parents=True, exist_ok=True)
    for i, pred in enumerate(preds_uncertainty):
        ax[i].plot(preds_regressor, pred, 'p')
    fig.suptitle('{} training points'.format(num), fontsize=16)
    if save_fig:
        fig.savefig(f'{folder_name}/experiment_{num}/samples_{num}_{idx}.png')