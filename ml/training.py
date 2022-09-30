import pickle
import keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from utils import config
from transform import moving_average_1d


class RANS_1DCNN:

    def __init__(self, rans, dns):
        self.model = self._create_model()
        self.rans_train, self.rans_validate, self.dns_train, self.dns_validate = train_test_split(rans, dns, test_size=0.2)

    def _create_model(self):
        model = keras.Sequential(
            [
                keras.Input((config.rans_layer_size,1)),
                keras.layers.Conv1D(config.dns_layer_size+4, config.kernel_size, 1),
                keras.layers.Dense(config.dns_layer_size+4),
                keras.layers.Dense(config.dns_layer_size+4),
                keras.layers.MaxPooling1D(config.kernel_size+4),
                keras.layers.Dense(config.dns_layer_size),
            ]
        )

        print(model.summary())
        return model

    def compile(self):
        self.model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['mean_absolute_percentage_error'],
        )
        return self

    def train(self, train_config, user_early_stop=False, use_tensorboard=True):

        callbacks = []

        callback_early_stopping_val_accuracy = tf.keras.callbacks.EarlyStopping(
            monitor="mean_absolute_percentage_error",
            baseline=350,
            mode='min',
            patience=1_000,
            restore_best_weights=True,
        )

        if user_early_stop:
            callbacks.append(callback_early_stopping_val_accuracy)

        if use_tensorboard:
            logdir = os.path.join("logs", datetime.datetime.now().strftime('%Y%m%dT%H%M%S%z'))
            callbacks.append(
                tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
            )

        self.model.fit(
            self.rans_train, self.dns_train,
            validation_data=(self.rans_validate, self.dns_validate),
            validation_freq=1,
            callbacks=callbacks,
            **train_config)

        return self

    def _vis_dataset(self, ax, ds, smooth_ma_window_size=None):
        # plt.figure()
        x, (rans, dns, ys, df) = ds
        pred_dns = self.model.predict(rans).squeeze()
        pred_scaled = pred_dns / 1_000_000
        dns_scaled = dns / 1_000_000
        if smooth_ma_window_size:
            pred_scaled = moving_average_1d(pred_scaled.reshape(-1), smooth_ma_window_size)
            dns_scaled = moving_average_1d(dns_scaled.reshape(-1), smooth_ma_window_size)
        ax.plot(ys, dns_scaled, label='dns')
        ax.plot(ys, pred_scaled, label='pred_dns')
        ax.legend()
        ax.title.set_text(x)

    def vis_pred(self, opath_fig):
        fig, axs = plt.subplots(ncols=6, nrows=6, figsize=(25,20), sharey=True)
        for ds, ax in zip(sorted(datalist, key=lambda x: x[0]), fig.get_axes()):
            self._vis_dataset(ax, ds, 6)
        for ax in fig.get_axes()[-4:]:
            ax.axis('off')
        plt.setp(axs, ylim=(0, 0.20))

        fig.savefig(opath_fig, facecolor='white', edgecolor='none', bbox_inches='tight')
        print(f'saved figure to {opath_fig}')
        return self

    def save(self, opath_model):
        self.model.save(opath_model)
        print(f'model saved at {opath_model}')
        return self


if __name__ == '__main__':
    with open('../data/rolling_windowed_ks.pickle', 'rb') as handle:
        dataset = pickle.load(handle)

    datalist = sorted(list(dataset.items()), key=lambda x: x[0])

    selected_datalist = [datalist[21], datalist[-3], datalist[-2]]

    rans = np.vstack([selected_datalist[i][1][0] for i in range(len(selected_datalist))])
    dns = np.vstack([selected_datalist[i][1][1] for i in range(len(selected_datalist))])

    oname_suffix = 'Cov1d-2FF-MaxPool-trained-with-0:1n7:8n-3-400'
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S%z')
    opath_model = os.path.join("saved_models", f'{timestamp}-{oname_suffix}')

    opath_fig = os.path.join(opath_model, 'rans-predicted-dns.pdf')

    train_config = dict(
        epochs=400,
        batch_size=5,
    )

    model = RANS_1DCNN(rans, dns).compile().train(train_config)
    model.vis_pred(opath_fig)
    model.save(opath_model)

