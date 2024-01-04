import pandas as pd
from tqdm import tqdm
import numpy as np
import polars as pl

class CFG:
    shift = 12
    seq_length = 360
    feature = 2
    train_ratio = 0.8
    batch_size = 32
    input_dim = 2
    embed_dim = 32
    num_classes = 1 
    num_layers = 4
    nhead = 4
    dim_feedforward = 64
    learning_rate = 1e-3
    epochs = 20
    train_record_steps = 80
    test_record_steps = 20
    dropout = 0.4
    scheduler_step_size = 30
    scheduler_gamma = 0.8
    
def make_train_dataset(train_data, train_events, drop_nulls=False):

    series_ids = train_data['series_id'].unique(maintain_order=True).to_list()
    X, y = pl.DataFrame(), pl.DataFrame()
    for idx in tqdm(series_ids):

        # Normalizing sample features
        sample = train_data.filter(pl.col('series_id') == idx).with_columns(
            [(pl.col(col) / pl.col(col).std()).cast(pl.Float32) for col in feature_cols if col != 'hour']
        )

        events = train_events.filter(pl.col('series_id') == idx)

        if drop_nulls:
            # Removing datapoints on dates where no data was recorded
            sample = sample.filter(
                pl.col('timestamp').dt.date().is_in(events['timestamp'].dt.date())
            )

        onsets = events.filter((pl.col('event') == 'onset') & (pl.col('step') != None))['step'].to_list()
        wakeups = events.filter((pl.col('event') == 'wakeup') & (pl.col('step') != None))['step'].to_list()

        asleep_column = sum([(onset <= pl.col('step')) & (pl.col('step') <= wakeup) for onset, wakeup in zip(onsets, wakeups)]).cast(pl.Boolean).alias('asleep')

        sample = sample.with_columns(asleep_column)
        X = X.vstack(sample[id_cols + feature_cols + ['asleep']])

    y = X.select('asleep').to_numpy().ravel()

    return X, y


train_events_path = './data/train_events.csv'
train_parquet_path = './data/train_series.parquet'
test_parquet_path = './data/test_series.parquet'

dt_transforms = [
    pl.col('timestamp').str.to_datetime(), 
    (pl.col('timestamp').str.to_datetime().dt.year()-2000).cast(pl.UInt8).alias('year'), 
    pl.col('timestamp').str.to_datetime().dt.month().cast(pl.UInt8).alias('month'),
    pl.col('timestamp').str.to_datetime().dt.day().cast(pl.UInt8).alias('day'), 
    pl.col('timestamp').str.to_datetime().dt.hour().cast(pl.UInt8).alias('hour')
]

data_transforms = [
    pl.col('anglez').cast(pl.Int16), # Casting anglez to 16 bit integer
    (pl.col('enmo')*1000).cast(pl.UInt16), # Convert enmo to 16 bit uint
]

train_series = pl.scan_parquet(train_parquet_path).with_columns(
    dt_transforms + data_transforms
    )

train_events = pl.read_csv(train_events_path).with_columns(
    dt_transforms
    ).drop_nulls()

mismatches = train_events.drop_nulls().group_by(['series_id', 'night']).agg([
    ((pl.col('event') == 'onset').sum() == (pl.col('event') == 'wakeup').sum()).alias('balanced')
    ]).sort(by=['series_id', 'night']).filter(~pl.col('balanced'))

unbalanced_data = pl.DataFrame()

for mm in mismatches.to_numpy(): 
    unbalanced_rows = train_events.filter((pl.col('series_id') == mm[0]) & (pl.col('night') == mm[1]))
    unbalanced_data = unbalanced_data.vstack(unbalanced_rows)
    train_events = train_events.filter(~((pl.col('series_id') == mm[0]) & (pl.col('night') == mm[1])))
    
# Getting series ids as a list for convenience
series_ids = train_events['series_id'].unique(maintain_order=True).to_list()

# Updating train_series to only keep these series ids
train_series = train_series.filter(pl.col('series_id').is_in(series_ids))

features, feature_cols = [pl.col('hour')], ['hour','anglez','enmo']

id_cols = ['series_id', 'step', 'timestamp']

X_train, y_train = make_train_dataset(train_series.collect(), train_events)
train = X_train.to_pandas()
train.asleep = train.asleep.apply(lambda x:1 if x else 0)


from sklearn.model_selection import (
    KFold, StratifiedKFold, StratifiedGroupKFold, train_test_split
)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(kf.split(series_ids)):
    if fold == 5:
        id_train = [series_ids[i] for i in train_idx]
        id_val = [series_ids[i] for i in valid_idx]
        
        
def preproces(data_id):
    '''
    data_id ：对应的id
    '''
    
    train_array = [] # 训练数据
    target_array = []

    target_cols = ['asleep']
    
    seq_length = 100
    cols = ['anglez','enmo']
    shift = CFG.shift
    seq_length = CFG.seq_length
    
    for i in tqdm(data_id):

        tmp_train = train.loc[train['series_id']==i,cols].values
        tmp_target = train.loc[train['series_id']==i,target_cols].values

        batch = (len(tmp_train))//shift
        print(f"batch is {batch}")
        print(f"length is {len(tmp_train)}")
        train_array_ = np.zeros([batch,seq_length,len(cols)])
        target_array_ = np.zeros([batch,seq_length,len(target_cols)])

        for b in tqdm(range(batch)):
            if b == batch-1:
                num_ = tmp_train[b * shift:]
                train_array_[b,:len(num_),:] = num_

                target_ = tmp_target[b * shift]
                target_array_[b,:len(target_),:] = target_
            
            elif b == 0:
                num_ = tmp_train[0:seq_length]
                train_array_[b,:,:] = num_

                target_ = tmp_target[0:seq_length]
                target_array_[b,:,:] = target_

            else :
                if(b * shift + seq_length>len(tmp_train)):
                    num_ = tmp_train[b * shift:b * shift + seq_length]
                    print(num_.shape)
                    train_array_[b,:len(num_),:] = num_

                    target_ = tmp_target[b * shift:b * shift + seq_length]
                    target_array_[b,:len(target_),:] = target_
                else:
                    num_ = tmp_train[b * shift:b * shift + seq_length]
                    train_array_[b,:,:] = num_

                    target_ = tmp_target[b * shift:b * shift + seq_length]
                    target_array_[b,:,:] = target_
        train_array.append(train_array_)
        target_array.append(target_array_)
        
    train_array = np.concatenate(train_array, axis=0)
    target_array = np.concatenate(target_array, axis=0)
    return train_array, target_array
print(len(id_train))
print(len(id_val))
train_array, target_array = preproces(id_train)
valid_array, traget_valid_array = preproces(id_val)
np.save('./data/train_array.npy',train_array)
np.save('./data/target_array.npy',train_array)
np.save('./data/valid_array.npy',train_array)
np.save('./data/target_valid_array.npy',train_array)

#python data_preprocess.py > data_preprocess.log 2>&1
#tail -f data_preprocess.log    
