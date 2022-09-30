import pickle
from tensorflow import keras
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from utils import config

# colorblind friendly
YELLOW, BLUE, GREEN, RED = "#E69F00", "#0072B2", "#009E73", "#CC79A7"
DNS_COLOR = GREEN
PRED_COLOR = YELLOW
RANS_COLOR = BLUE


def moving_average_1d(a, window_size):
    assert window_size > 0, 'window_size needs to be positive'
    return np.convolve(np.pad(a, window_size-1, mode='edge'), np.ones(window_size), 'valid')[window_size-1:window_size-1+len(a)] / window_size


def viz_dataset(ax, ds, smooth_ma_window_size=None):
    # plt.figure()
    x, (rans, dns, ys, df) = ds
    pred_dns = model.predict(rans).squeeze()
    pred_scaled = pred_dns / 1_000_000
    dns_scaled = dns / 1_000_000
    if smooth_ma_window_size:
        pred_scaled = moving_average_1d(pred_scaled.reshape(-1), smooth_ma_window_size)
        dns_scaled = moving_average_1d(dns_scaled.reshape(-1), smooth_ma_window_size)
    ax.plot(ys, dns_scaled, label='dns', color=DNS_COLOR)
    ax.plot(ys, pred_scaled, label='pred', color=PRED_COLOR)
    ax.legend()
    ax.title.set_text(f"$x/c = {x}$")


def viz_loss(ax, ds, smooth_ma_window_size=None, has_title=True):
    # plt.figure()
    x, (rans, dns, ys, df) = ds
    pred_dns = model.predict(rans).squeeze()
    pred_scaled = pred_dns / 1_000_000
    dns_scaled = dns / 1_000_000
    rans_scaled = rans / 1_000_000
    if smooth_ma_window_size:
        pred_scaled = moving_average_1d(pred_scaled.reshape(-1), smooth_ma_window_size)
        dns_scaled = moving_average_1d(dns_scaled.reshape(-1), smooth_ma_window_size)
        rans_scaled = rans_scaled[:, 0].reshape(-1)
    ax.plot(ys, np.abs(dns_scaled - rans_scaled), label=r'$L_c^1(\texttt{rans})$', color=RANS_COLOR)
    ax.plot(ys, np.abs(dns_scaled - pred_scaled), label=r'$L_c^1(\texttt{pred})$', color=PRED_COLOR)
    ax.set_yscale('log')
    ax.legend()
    if has_title:
        ax.title.set_text(f"$x/c = {x}$")


def plot_viz(datalist):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 14
    fig = plt.figure(constrained_layout=True, tight_layout=False, figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)
    axs = []
    for ds, g in zip(sorted(datalist, key=lambda x: x[0]), gs):
        ax = fig.add_subplot(g)
        axs.append(ax)
        viz_dataset(ax, ds, 6)
    plt.setp(axs, ylim=(0, 0.20))
    fig.supxlabel(f"$y$")
    fig.supylabel(f"$CF_k$")
    return fig


def plot_loss(datalist):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 14
    fig = plt.figure(constrained_layout=True, tight_layout=False, figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)
    axs = []
    for ds, g in zip(sorted(datalist, key=lambda x: x[0]), gs):
        ax = fig.add_subplot(g)
        axs.append(ax)
        viz_loss(ax, ds, 6)
    plt.setp(axs, ylim=(1e-5, 1e1))
    fig.supxlabel(r"$y$")
    fig.supylabel(r"$L_c^{1}(CF_k)$")
    return fig


def plot_viz_loss(datalist):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 18
    fig = plt.figure(constrained_layout=True, tight_layout=False, figsize=(16, 8))
    gs = fig.add_gridspec(2, 4)
    gsa= np.array(list(gs)).reshape(2, -1)
    axs_viz, axs_loss = [], []
    for ds, g in zip(sorted(datalist, key=lambda x: x[0]), gsa[0, :]):
        ax = fig.add_subplot(g)
        axs_viz.append(ax)
        viz_dataset(ax, ds, 6)
    axs_viz[0].set_ylabel(r"$CF_k$")
    plt.setp(axs_viz, ylim=(0, 0.20))
    for ds, g in zip(sorted(datalist, key=lambda x: x[0]), gsa[1, :]):
        ax = fig.add_subplot(g)
        axs_loss.append(ax)
        viz_loss(ax, ds, 6, has_title=False)
    axs_loss[0].set_ylabel(r"$L_c^{1}(CF_k)$")
    plt.setp(axs_loss, ylim=(1e-5, 1e1))
    fig.supxlabel(r"$y$")
    return fig


def save_figure(fig, opath_fig):
    fig.savefig(opath_fig, facecolor='white', edgecolor='none', bbox_inches='tight')
    print(f'saved figure to {opath_fig}')


if __name__ == '__main__':
    NORMALIZED_DATASET_PICKLE = '../data/rolling_windowed_ks.pickle'
    MODEL_PATH = "../ml/saved_models/20220918T140937-Cov1d-2FF-MaxPool-trained-with-0:1n7:8n-3-400"
    # SELECTED_X_POSITIONS = [0.14, 0.15, 0.16, 0.17]
    # SELECTED_X_POSITIONS = [0.19, 0.25, 0.32, 0.44]
    SELECTED_X_POSITIONS = [0.17, 0.25, 0.32, 0.44]

    with open(NORMALIZED_DATASET_PICKLE, 'rb') as handle:
        dataset = pickle.load(handle)

    print(sorted(dataset.keys()))
    # exit(0)

    model = keras.models.load_model(MODEL_PATH)

    datalist = [(x, dataset[x]) for x in SELECTED_X_POSITIONS]

    # opath_viz = os.path.join('../fig', 'rans-predicted-dns-main4-viz.pdf')
    # opath_loss = os.path.join('../fig', 'rans-predicted-dns-main4-loss.pdf')
    # opath_viz_loss = os.path.join('../fig', 'rans-predicted-dns-main4-viz-loss-earlier.pdf')
    opath_viz_loss = os.path.join('../fig', 'rans-predicted-dns-main4-viz-loss.pdf')

    # fig_viz = plot_viz(datalist)
    # plt.show()
    # save_figure(fig_viz, opath_viz)
    #
    # fig_loss = plot_loss(datalist)
    # plt.show()
    # save_figure(fig_loss, opath_loss)

    fig_viz_loss = plot_viz_loss(datalist)
    plt.show()
    save_figure(fig_viz_loss, opath_viz_loss)
