from math import log

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def get_corr_matrix(df):
    return df.corr(min_periods=1)


def delete_nan_cols(df):
    # series with column's nan count
    nan_count_series = df.isna().sum()
    column_size = df.shape[0]

    for i, v in nan_count_series.iteritems():
        if (v / column_size) * 100 > 60 and i != "G_total":
            df.drop(i, axis=1, inplace=True)

    return df


def delete_extreme_vals(df):
    for i in ['Рлин', 'Рлин_1', 'Туст', 'Тна шлейфе', 'Тзаб', 'Дебит ст. конд.', 'Дебит кон нестабильный',
              'Рпл. Тек (Карноухов)', 'Удельная плотность газа ']:
        vals = df[i].to_numpy()
        q25 = np.nanquantile(vals, 0.25)
        q75 = np.nanquantile(vals, 0.75)
        low = (q25 - 1.5 * (q75 - q25))
        high = (q75 + 1.5 * (q75 - q25))

        df = df.loc[~((low > df[i]) | (df[i] > high)), :]
        # vals1 = df[i].to_numpy()
        # sns.displot(vals1, kde=True, bins=int(1+log(len(vals1), 2)))
        # plt.title(f'{i}')
        # plt.savefig(i + '.jpg')

    return df


def delete_small_uniques(df):
    del_list = []
    for i in list(df):
        if df[i].nunique(dropna=True) == 1:
            del_list.append(i)

    for i in del_list:
        df.drop(i, axis=1, inplace=True)

    return df


def parse_xlsx() -> pd.DataFrame:
    df = pd.read_csv('data/clean.csv')

    return df


def clear_dataframe(df):
    # delete empty samples with empty classes
    df.dropna(subset=["КГФ", "G_total"], how='all', inplace=True)

    df = delete_extreme_vals(df)
    df = delete_small_uniques(df)
    df = delete_nan_cols(df)

    return df


def get_class_list(df):
    kgf = sorted(df[df.columns[-1]].unique())

    bins = 1 + log(len(kgf), 2)
    min = kgf[0]
    max = kgf[len(kgf) - 1]
    length = (max - min) / bins

    kgf_int = [(min + length * i, min + length * (i+1)) for i in range(int(bins))]
    classes = set()

    for row in df.iterrows():
        g_val = row[:-2]
        kgf_val = row[:-1]

        for interval in kgf_int:
            print(kgf_val, ' ', interval)
            if interval[0] <= kgf_val < interval[1]:
                classes.add((g_val, interval))
                break

    return classes


def main() -> None:
    data = parse_xlsx()
    data = clear_dataframe(data)

    corr = get_corr_matrix(data)
    sns.heatmap(corr, xticklabels=data.shape[1], yticklabels=1, linewidths=.5)
    plt.savefig('heatmap.jpg')

    print(get_class_list(data))



if __name__ == '__main__':
    main()
