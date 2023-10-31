import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
import datetime
from tqdm import tqdm

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from metric import score  # Import event detection ap score function

# These are variables to be used by the score function
column_names = {
    'series_id_column_name': 'series_id',
    'time_column_name': 'step',
    'event_column_name': 'event',
    'score_column_name': 'score',
}

tolerances = {
    'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
    'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
}

# Importing data

# Column transformations

train_events_path = '../data/train_events.csv'
train_parquet_path = '../data/train_series.parquet'
test_parquet_path = '../data/test_series.parquet'

dt_transforms = [
    pl.col('timestamp').str.to_datetime(),
    (pl.col('timestamp').str.to_datetime().dt.year() -
     2000).cast(pl.UInt8).alias('year'),
    pl.col('timestamp').str.to_datetime().dt.month().cast(
        pl.UInt8).alias('month'),
    pl.col('timestamp').str.to_datetime().dt.day().cast(pl.UInt8).alias('day'),
    pl.col('timestamp').str.to_datetime(
    ).dt.hour().cast(pl.UInt8).alias('hour')
]

data_transforms = [
    pl.col('anglez').cast(pl.Int16),  # Casting anglez to 16 bit integer
    (pl.col('enmo')*1000).cast(pl.UInt16),  # Convert enmo to 16 bit uint
]

train_series = pl.scan_parquet(train_parquet_path).with_columns(
    dt_transforms + data_transforms
)

train_events = pl.read_csv(train_events_path).with_columns(
    dt_transforms
).drop_nulls()

test_series = pl.scan_parquet(test_parquet_path).with_columns(
    dt_transforms + data_transforms
)

# Removing null events and nights with mismatched counts from series_events
# 去除空值和当晚起床次数和睡眠次数不相符的
# 等于是只有睡觉没有起床，或者相反
# 有五个
mismatches = train_events.drop_nulls().group_by(['series_id', 'night']).agg([
    ((pl.col('event') == 'onset').sum() ==
     (pl.col('event') == 'wakeup').sum()).alias('balanced')
]).sort(by=['series_id', 'night']).filter(~pl.col('balanced'))

unbalanced_data = pl.DataFrame()

for mm in mismatches.to_numpy():
    unbalanced_rows = train_events.filter(
        (pl.col('series_id') == mm[0]) & (pl.col('night') == mm[1]))
    unbalanced_data = unbalanced_data.vstack(unbalanced_rows)
    train_events = train_events.filter(
        ~((pl.col('series_id') == mm[0]) & (pl.col('night') == mm[1])))

# Getting series ids as a list for convenience
series_ids = train_events['series_id'].unique(maintain_order=True).to_list()

# Updating train_series to only keep these series ids
train_series = train_series.filter(pl.col('series_id').is_in(series_ids))

features, feature_cols = [pl.col('hour')], ['hour']

for mins in [10, 30, 60, 60*3]:

    for var in ['anglez', 'enmo']:
        # 滚动窗口求值
        features += [
            pl.col(var).rolling_mean(mins, center=True, min_periods=1).abs().cast(
                pl.UInt16).alias(f'{var}_{mins}m_mean'),
            pl.col(var).rolling_std(mins, center=True, min_periods=1).abs().cast(
                pl.UInt16).alias(f'{var}_{mins}m_std'),
            (pl.col(var).rolling_max(mins, center=True, min_periods=1) -
             pl.col(var).rolling_min(mins, center=True, min_periods=1))
            .abs().cast(pl.UInt32).alias(f'{var}_{mins}m_range')
        ]

        feature_cols += [
            f'{var}_{mins}m_mean', f'{var}_{mins}m_std', f'{var}_{mins}m_range'
        ]

        # Getting first variations
        # 差分后求值，乘以10
        # 等于是 v(t+mins) - v(t)
        features += [
            (pl.col(var).diff().abs().rolling_mean(mins, center=True, min_periods=1)
             * 10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_mean'),
            (pl.col(var).diff().abs().rolling_std(mins, center=True, min_periods=1)
             * 10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_std'),
            (pl.col(var).diff().abs().rolling_max(mins, center=True, min_periods=1) -
             pl.col(var).diff().abs().rolling_min(mins, center=True, min_periods=1))
            .abs().cast(pl.UInt32).alias(f'{var}_{mins}m_diff_range')
        ]

        feature_cols += [
            f'{var}_1v_{mins}m_mean',  f'{var}_1v_{mins}m_std', f'{var}_{mins}m_diff_range'
        ]

        # 二阶差分
        features += [
            (pl.col(var).diff().diff().rolling_mean(mins, center=True, min_periods=1)
             * 10).abs().cast(pl.UInt32).alias(f'{var}_2v_{mins}m_mean'),
            (pl.col(var).diff().diff().rolling_std(mins, center=True, min_periods=1)
             * 10).abs().cast(pl.UInt32).alias(f'{var}_2v_{mins}m_std'),
        ]

        feature_cols += [
            f'{var}_2v_{mins}m_mean', f'{var}_2v_{mins}m_std'
        ]


id_cols = ['series_id', 'step', 'timestamp']

train_series = train_series.with_columns(
    features
).select(id_cols + feature_cols)

test_series = test_series.with_columns(
    features
).select(id_cols + feature_cols)


def make_train_dataset(train_data, train_events, drop_nulls=False):

    series_ids = train_data['series_id'].unique(maintain_order=True).to_list()
    X, y = pl.DataFrame(), pl.DataFrame()
    for idx in tqdm(series_ids):

        # Normalizing sample features
        sample = train_data.filter(pl.col('series_id') == idx).with_columns(
            [(pl.col(col) / pl.col(col).std()).cast(pl.Float32)
             for col in feature_cols if col != 'hour']
        )

        events = train_events.filter(pl.col('series_id') == idx)

        if drop_nulls:
            # Removing datapoints on dates where no data was recorded
            sample = sample.filter(
                pl.col('timestamp').dt.date().is_in(
                    events['timestamp'].dt.date())
            )

        onsets = events.filter((pl.col('event') == 'onset') & (
            pl.col('step') != None))['step'].to_list()
        wakeups = events.filter((pl.col('event') == 'wakeup') & (
            pl.col('step') != None))['step'].to_list()

        asleep_column = sum([(onset <= pl.col('step')) & (pl.col('step') <= wakeup)
                            for onset, wakeup in zip(onsets, wakeups)]).cast(pl.Boolean).alias('asleep')

        sample = sample.with_columns(asleep_column)
        X = X.vstack(sample[id_cols + feature_cols + ['asleep']])

    y = X.select('asleep').to_numpy().ravel()

    return X, y


def get_events(series, classifier):
    '''
    Takes a time series and a classifier and returns a formatted submission dataframe.
    '''

    series_ids = series['series_id'].unique(maintain_order=True).to_list()
    events = pl.DataFrame(
        schema={'series_id': str, 'step': int, 'event': str, 'score': float})

    for idx in tqdm(series_ids):

        # Collecting sample and normalizing features
        scale_cols = [col for col in feature_cols if (
            col != 'hour') & (series[col].std() != 0)]
        X = series.filter(pl.col('series_id') == idx).select(id_cols + feature_cols).with_columns(
            [(pl.col(col) / series[col].std()).cast(pl.Float32)
             for col in scale_cols]
        )

        # Applying classifier to get predictions and scores
        preds, probs = classifier.predict(
            X[feature_cols]), classifier.predict_proba(X[feature_cols])[:, 1]

        # NOTE: Considered using rolling max to get sleep periods excluding <30 min interruptions, but ended up decreasing performance
        X = X.with_columns(
            pl.lit(preds).cast(pl.Int8).alias('prediction'),
            pl.lit(probs).alias('probability')
        )

        # Getting predicted onset and wakeup time steps
        # 表示状态有变化的step
        pred_onsets = X.filter(X['prediction'].diff() > 0)['step'].to_list()
        pred_wakeups = X.filter(X['prediction'].diff() < 0)['step'].to_list()

        if len(pred_onsets) > 0:

            # Ensuring all predicted sleep periods begin and end
            if min(pred_wakeups) < min(pred_onsets):
                pred_wakeups = pred_wakeups[1:]

            if max(pred_onsets) > max(pred_wakeups):
                pred_onsets = pred_onsets[:-1]

            # Keeping sleep periods longer than 30 minutes
            sleep_periods = [(onset, wakeup) for onset, wakeup in zip(
                pred_onsets, pred_wakeups) if wakeup - onset >= 30]

            for onset, wakeup in sleep_periods:
                # Scoring using mean probability over period
                score = X.filter((pl.col('step') >= onset) & (
                    pl.col('step') <= wakeup))['probability'].mean()

                # Adding sleep event to dataframe
                events = events.vstack(pl.DataFrame().with_columns(
                    pl.Series([idx, idx]).alias('series_id'),
                    pl.Series([onset, wakeup]).alias('step'),
                    pl.Series(['onset', 'wakeup']).alias('event'),
                    pl.Series([score, score]).alias('score')
                ))

    # Adding row id column
    events = events.to_pandas().reset_index().rename(
        columns={'index': 'row_id'})

    return events


train_data = train_series.filter(pl.col('series_id').is_in(
    series_ids)).take_every(5).collect()  # .take_every(12 * 5)
# Creating train dataset
X_train, y_train = make_train_dataset(train_data, train_events)
