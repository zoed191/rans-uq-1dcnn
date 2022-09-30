import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from os import path
import pickle
from utils import config


class RollingWindowGenerator:
    def __init__(self, dns_csv, rans_csv):
        self.dns = pd.read_csv(dns_csv)
        self.rans = pd.read_csv(rans_csv)

    def with_window_sizes(self, dns_window_size, rans_window_size):
        self.dns_window_size = dns_window_size
        self.rans_window_size = rans_window_size
        assert self.rans_window_size >= self.dns_window_size, "RANS's window size must not less than DNS's widnow size"
        return self

    def _get_rolling_windowed_rans(self, df_rans):
        ks = df_rans['k'].to_numpy() * 1_000_000
        # offset = round((self.rans_window_size - self.dns_window_size) / 2)
        # return np.diff(sliding_window_view(ks, self.rans_window_size)[:-offset])
        # return sliding_window_view(ks, self.rans_window_size)[:-offset]
        return sliding_window_view(ks, self.rans_window_size)

    def _get_rolling_windowed_dns(self, df_dns):
        ks = df_dns['k'].to_numpy() * 1_000_000
        # offset = round((self.rans_window_size - self.dns_window_size) / 2)
        # return np.diff(sliding_window_view(ks, self.dns_window_size)[offset:])
        # return sliding_window_view(ks, self.dns_window_size)[offset:]
        return sliding_window_view(ks, self.dns_window_size)

    def _get_rolling_windowed_ys(self, df_dns):
        ks = df_dns['y'].to_numpy()
        # offset = round((self.rans_window_size - self.dns_window_size) / 2)
        # return sliding_window_view(ks, self.dns_window_size)[offset:]
        return sliding_window_view(ks, self.dns_window_size)

    def _validate(self, dataset):
        for x, (rans, dns, ys, *_) in dataset.items():
            assert len(rans) == len(dns) == len(ys)

    def to_pickle(self, output_path):
        dataset = {}
        dns_grp, rans_grp = self.dns.groupby('x'), self.rans.groupby('x')
        for x in dns_grp.groups.keys() & rans_grp.groups.keys():
            dns_ks = self._get_rolling_windowed_dns(dns_grp.get_group(x))
            dns_ys = self._get_rolling_windowed_ys(dns_grp.get_group(x))
            dns_df = dns_grp.get_group(x)
            rans_ks = self._get_rolling_windowed_rans(rans_grp.get_group(x))
            min_len = min(len(dns_ks), len(rans_ks))
            dataset[x] = rans_ks[:min_len], dns_ks[:min_len], dns_ys[:min_len], dns_df
        self._validate(dataset)
        with open(output_path, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    path_dns = '../data/normalized_dns_in_range.csv'
    path_rns = '../data/normalized_rans_in_range.csv'
    rwg = RollingWindowGenerator(path_dns, path_rns)
    rwg.with_window_sizes(config.dns_window_size, config.rans_window_size).to_pickle('../data/rolling_windowed_ks.pickle')
