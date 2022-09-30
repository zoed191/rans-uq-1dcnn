import pickle
from tensorflow import keras
import os
from transform import moving_average_1d

from preprocessing.normalize_rawdata import denormalize_dns_dataframe
from preprocessing.normalize_rawdata import denormalize_rans_dataframe
from utils import config


def export_dataset(ds, smooth_ma_window_size=None):
    # plt.figure()
    x, (rans, dns, ys, df) = ds
    pred_dns = model.predict(rans).squeeze()
    pred_scaled = pred_dns / 1_000_000
    dns_scaled = dns / 1_000_000
    if smooth_ma_window_size:
        pred_scaled = moving_average_1d(pred_scaled.reshape(-1), smooth_ma_window_size)
        dns_scaled = moving_average_1d(dns_scaled.reshape(-1), smooth_ma_window_size)

    df_pred_rans = denormalize_rans_dataframe(x, ys, pred_scaled)
    opath_pred_rans = os.path.join(config.opath_model, 'prediction', f'E_CDFkMean_x{round(x*100):04d}_CNN_DNS.csv')
    df_pred_rans.to_csv(opath_pred_rans, index=False)
    print(f'write to {opath_pred_rans}')
    df_dns = denormalize_dns_dataframe(x, ys, df['k'].to_numpy()[:len(ys)])
    opath_dns = os.path.join(config.opath_model, 'prediction', f'B_DNS_x{round(x*100):03d}.csv')
    df_dns.to_csv(opath_dns, index=False)
    print(f'write to {opath_dns}')


if __name__ == '__main__':
    model = keras.models.load_model("saved_models/20220918T140937-Cov1d-2FF-MaxPool-trained-with-0:1n7:8n-3-400")

    with open('../data/rolling_windowed_ks.pickle', 'rb') as handle:
        dataset = pickle.load(handle)

    datalist = sorted(list(dataset.items()), key=lambda x: x[0])

    for ds in datalist:
        export_dataset(ds, 6)