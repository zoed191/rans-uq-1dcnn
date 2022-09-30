import pandas as pd
from typing import Callable
import os
import re
import numpy as np

NormalizeFunction = Callable[[pd.DataFrame], pd.DataFrame]


def normalize_rans_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    colname_mapping = {
        'Points:0': 'x',
        'Points:1': 'y',
        'kMean': 'k',
    }
    normalize_mapping = {
        'x': lambda x: x / 0.2,
        'y': lambda y: y / 0.2,
        'k': lambda k: k / 4.5**2,
    }
    return df.drop(columns=["Points:2"]).rename(columns=colname_mapping).transform(normalize_mapping)


def denormalize_rans_dataframe(x: float, y: np.array, k: np.array) -> pd.DataFrame:
    xs = np.ones(y.shape) * x * 0.2
    ys = y * 0.2
    ks = k * 4.5**2
    df = pd.DataFrame({
        'Points:0': xs.reshape(-1),
        'Points:1': ys.reshape(-1),
        'Points:2': xs.reshape(-1),
        'kMean': ks.reshape(-1)
    })
    return df


def normalize_dns_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    colname_mapping = {
        'x_c': 'x',
        'y_c': 'y',
        'k_u2': 'k',
    }
    selected_colname = ['x', 'y', 'k']
    normalize_mapping = {
        'x': lambda x: x / 1,
        'y': lambda y: y / 1,
        'k': lambda k: k / 0.2**2,
    }
    return df.rename(columns=colname_mapping)[selected_colname].transform(normalize_mapping)


def denormalize_dns_dataframe(x: float, y: np.array, k_raw: np.array) -> pd.DataFrame:
    xs = np.ones(y.shape) * x
    ys = y.reshape(-1)
    # ks = k * 0.2**2
    ks = k_raw.reshape(-1)
    df = pd.DataFrame({
        'x_c': xs.reshape(-1),
        'y_c': ys.reshape(-1),
        'k_u2': ks.reshape(-1)
    })
    return df


def preprocess_raw_data(
        input_folder: str,
        input_filename_template: str,
        output_csv_path: str,
        normalize_function: NormalizeFunction,
        save_raw:bool=True,
) -> pd.DataFrame:
    """
    Preprocesses raw data in csvs.
    :param input_folder:  path to input directory
    :param input_filename_template: regex of the input filename
    :param output_csv_path: path to output csv
    :param normalize_function: normalization function to apply to read file
    :return: preprocessed data
    """
    input_fpaths = (f for f in os.listdir(input_folder) if re.match(input_filename_template, f))
    df_raw = pd.concat(
            map(pd.read_csv,
                map(lambda f: os.path.join(INPUT_FOLDER, f),
                    sorted(input_fpaths))),
    )
    df = normalize_function(df_raw).groupby(['x', 'y']).median().reset_index()
    if save_raw:
        df_raw.to_csv(output_csv_path.replace('.csv', '_raw.csv'), index=False, float_format='%g')
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False, float_format='%g')
    return df


if __name__ == '__main__':
    INPUT_FOLDER = '../data/k_forMarker'
    OUTPUT_FOLDER = '../data/preprocessed_data'

    config_rans = dict(
        input_folder=INPUT_FOLDER,
        input_filename_template=r'A_CFDkMean_.*.csv',
        output_csv_path=os.path.join(OUTPUT_FOLDER, 'rans.csv'),
        normalize_function=normalize_rans_dataframe,
    )

    config_dns = dict(
        input_folder=INPUT_FOLDER,
        input_filename_template=r'B_DNS_.*.csv',
        output_csv_path=os.path.join(OUTPUT_FOLDER, 'dns.csv'),
        normalize_function=normalize_dns_dataframe,
    )

    for cfg in [config_dns, config_rans]:
        preprocess_raw_data(**cfg)
