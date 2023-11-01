import pandas as pd
from tqdm import tqdm
import numpy as np
import polars as pl
import os
save_dir = "predict_step"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class CFG:
    shift = 60
    seq_length = 360
    feature = 2
    batch_size = 256
    input_dim = 2
    embed_dim = 32
    num_classes = 1 
    num_layers = 4
    nhead = 4
    dim_feedforward = 64
    learning_rate = 1e-3
    epochs = 20
    train_record_steps = 500
    test_record_steps = 500
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
        
        # step_to_night = events.filter(pl.col('step') != None).select(['step', 'night']).to_pandas().set_index('step')['night'].to_dict()
        # sample = sample.with_columns(
        #     pl.col('step').map_batches(step_to_night).alias('night')
        # )
        
        if drop_nulls:
            # Removing datapoints on dates where no data was recorded
            sample = sample.filter(
                pl.col('timestamp').dt.date().is_in(events['timestamp'].dt.date())
            )

        onsets = events.filter((pl.col('event') == 'onset') & (pl.col('step') != None))['step'].to_list()
        wakeups = events.filter((pl.col('event') == 'wakeup') & (pl.col('step') != None))['step'].to_list()
        
        trans_conditions = pl.lit(False)
        for onset, wakeup in zip(onsets, wakeups):
            trans_conditions = trans_conditions | ((pl.col('step') == onset) | (pl.col('step') == wakeup))

        trans_column = trans_conditions.alias('trans')
        
        #asleep_column = sum([(onset <= pl.col('step')) & (pl.col('step') <= wakeup) for onset, wakeup in zip(onsets, wakeups)]).cast(pl.Boolean).alias('asleep')
        
        #trans_column = [(onset == pl.col('step')) | (pl.col('step') == wakeup) for onset, wakeup in zip(onsets, wakeups)].alias('trans')
        
        #sample = sample.with_columns(asleep_column)
        sample = sample.with_columns(trans_column)
        
        X = X.vstack(sample[id_cols + feature_cols + ['trans']])

    #y = X.select('asleep').to_numpy().ravel()

    return X

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

X_train = make_train_dataset(train_series.collect(), train_events)
train = X_train.to_pandas()
del X_train

#train.to_parquet('./data/after_create_feature.parquet',index = False)

train_events = pd.read_csv('./data/train_events.csv')
train_events = train_events.dropna()
df = train.merge(train_events.loc[:,['series_id','night','step']],on = ['series_id','step'], how = 'left')
# 首先找到所有trans为True的行的索引
trans_true_indices = df.index[df['trans']].tolist()

# 初始化一个新列来存储距离
df['distance_to_next_trans'] = 0
df['true_night'] = 0

# 标记正负号，初始为负，然后每次转换后都变号
sign = -1

# 上一个trans为True的索引，初始化为第一个索引之前的位置
last_trans_index = 0
# 遍历所有trans为True的索引
for trans_index in tqdm(trans_true_indices + [len(df)]):
    #print(last_trans_index,trans_index)
    # 计算距离范围的值
    if sign==-1:
        distance_values = list(range(-1 * (trans_index - last_trans_index), 0)) #最后一个值是-1
    else:
        distance_values = list(range((trans_index - last_trans_index),0,-1)) # 最后一个是1
        
    df.loc[last_trans_index :trans_index-1, 'distance_to_next_trans'] = distance_values    
    
    if trans_index == len(df):
        df.loc[last_trans_index :trans_index - 1, 'true_night'] = df.loc[last_trans_index-1, 'night'] + 1 #有点问题
    else:
        df.loc[last_trans_index :trans_index, 'true_night'] = df.loc[trans_index, 'night'] #往前赋值
    # 更新最后一个trans的索引 
    last_trans_index = trans_index + 1 #调过变化的点
    
    # 变换符号（负变正，正变负）
    sign *= -1
    
df = df.drop(columns = ['night','trans']).reset_index(drop = True)

def transform_distance(group):
    # 如果distance_to_next_trans全为0，则直接返回原组
    if (group['distance_to_next_trans'] == 0).all():
        return group
    
    # 找到非零值的最大绝对值
    max_abs_value = group.loc[group['distance_to_next_trans'] != 0, 'distance_to_next_trans'].abs().max()
    
    # 应用转换：非零值除以最大绝对值，零值保持不变
    group['distance_to_next_trans'] = group['distance_to_next_trans'].apply(
        lambda x: x / max_abs_value if x != 0 else 0
    )
    
    return group

# 对数据进行分组并应用转换
df_transformed = df.groupby(['series_id', 'true_night']).apply(transform_distance)

df_transformed = df_transformed.reset_index(drop = True)

train = df_transformed

from sklearn.model_selection import (
    KFold, StratifiedKFold, StratifiedGroupKFold, train_test_split
)
series_ids = list(train['series_id'].unique())
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(kf.split(series_ids)):
    if fold == 5:
        id_train = [series_ids[i] for i in train_idx]
        id_val = [series_ids[i] for i in valid_idx]
        
del df_transformed

def preproces(data_id):
    '''
    data_id ：对应的id
    '''
    
    train_array = [] # 训练数据
    target_array = []

    target_cols = ['distance_to_next_trans']
    
    seq_length = 100
    cols = ['anglez','enmo']
    shift = CFG.shift
    seq_length = CFG.seq_length
    
    for i in tqdm(data_id):

        tmp_train = train.loc[train['series_id']==i,cols].values
        tmp_target = train.loc[train['series_id']==i,target_cols].values

        batch = (len(tmp_train))//shift
        # print(f"batch is {batch}")
        # print(f"length is {len(tmp_train)}")
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
# np.save(f'./data/train_array_shift{CFG.shift}_seqLength{CFG.seq_length}.npy',train_array)
# np.save(f'./data/target_array_shift{CFG.shift}_seqLength{CFG.seq_length}.npy',target_array)
# np.save(f'./data/valid_array_shift{CFG.shift}_seqLength{CFG.seq_length}.npy',valid_array)
# np.save(f'./data/target_valid_array_shift{CFG.shift}_seqLength{CFG.seq_length}.npy',traget_valid_array)
# import os

# os.system("/usr/bin/shutdown")
#python data_preprocess.py > data_preprocess.log 2>&1
#tail -f data_preprocess.log    



target_array = target_array * 100
traget_valid_array = traget_valid_array * 100


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset,random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau,StepLR

def save_model(model, optimizer, filename="checkpoint.pth.tar"):
    filename = os.path.join(save_dir, filename)
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)

def load_model(model, optimizer, filename="checkpoint.pth.tar"):
    filename = os.path.join(save_dir, filename)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def validate_model(model, valid_loader, criterion, device):
    model.eval()  # Set model to evaluate mode
    test_loss = 0.0
    # 迭代验证集数据
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
    test_loss /= len(valid_loader)
    
    # 记录到TensorBoard
    model.train()  # Set model back to train mode
        
    return test_loss

class SleepDataset(Dataset):
    def __init__(
        self,
        train=None,
        y=None,
    ):
        self.train = train
        self.y = y

    def __len__(self):
        return len(self.train)

    def __getitem__(self, item):
        train = torch.tensor(self.train[item],dtype=torch.float32)
        y = torch.tensor(self.y[item],dtype=torch.float32)
        return train, y 

train_dataset = SleepDataset(train_array,target_array)
valid_dataset = SleepDataset(valid_array,traget_valid_array)

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=True)

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_layers=CFG.num_layers,
        nhead=4,
        dim_feedforward=64,
        num_classes = CFG.num_classes,
        dropout=0.4
    ):
        super(TransformerModel, self).__init__()

        # 1D convolution layer to transform each 2-dimensional feature into embed_dim dimensions
        self.conv1d = nn.Conv1d(
            in_channels=2,  # change this to 2 to match the last dimension of the input
            out_channels=embed_dim, 
            kernel_size=1,  # use kernel_size=1 because we want to apply the transformation to each time step independently
            padding=0  # no padding is needed with kernel_size=1
        )

        # Linear layer is not needed anymore as conv1d already does the job of dimension transformation
        # Layer normalization to stabilize the learning process
        self.layernorm = nn.LayerNorm(embed_dim)

        # Transformer Encoder to process the sequence data
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ),
            num_layers=num_layers
        )

        # Classification layer to reduce embed_dim dimensions to 1
        self.classification = nn.Linear(embed_dim, num_classes)  # output_dim is 1 as we want to output one value for each time step

    def forward(self, x):
        # Permute the input to have the correct order for Conv1d (batch_size, channels, seq_length)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)

        # Permute the output to have the correct order for TransformerEncoder (batch_size,seq_length, embed_dim)
        x = x.permute(0, 2, 1)

        # Apply layer normalization
        x = self.layernorm(x)

        # Pass the normalized embeddings through the transformer encoder
        x = self.encoder(x)

        # Reshape the output from the transformer to be suitable for the classification layer
        # Permute back to (batch_size, seq_length, embed_dim)
        x = x.permute(0, 1, 2)

        # Apply the classification layer to every time step
        x = self.classification(x)
        return x

# Instantiate the model with the desired parameters
input_dim = CFG.feature  
embed_dim = CFG.embed_dim 
model = TransformerModel(input_dim=input_dim, embed_dim=embed_dim)


import logging
from torch.utils.tensorboard import SummaryWriter

# 配置logging模块
logging.basicConfig(filename='training.log', level=logging.INFO)

# 创建TensorBoard summary writer
#writer = SummaryWriter('/root/tf-logs/')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
scheduler = StepLR(optimizer, step_size=CFG.scheduler_step_size, gamma=CFG.scheduler_gamma)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
logging.info(f"Total number of parameters: {total_params}")

model.train()
train_losses = []
train_step_losses = []
test_losses = []
test_step_losses = []
best_test_loss = float('inf')

last_step_model_path = ""
last_best_model_path = os.path.join(save_dir, "model_step_best.pth.tar")

for epoch in range(0, CFG.epochs):
    train_loss = 0.0
    test_loss = 0.0
    train_step_loss = 0.0
    test_step_loss = 0.0
    total_step = len(train_loader) * CFG.epochs
    
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

        now_step = epoch * (len(train_loader)) + batch_idx
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)

        loss = criterion(output, target) #可能全为0
        train_loss += loss.item()
        train_step_loss = loss.item()

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        if (now_step +1) % CFG.train_record_steps == 0:
            if last_step_model_path != "" and os.path.exists(last_step_model_path):
                os.remove(last_step_model_path)
                logging.info(f"Deleted last saved model at {last_step_model_path}")
                
            #train_step_loss /= CFG.train_record_steps
            train_step_losses.append(train_step_loss)

            # 记录到TensorBoard
            #writer.add_scalar('Training Loss', train_step_loss, now_step)

            logging.info(f"Epoch: {epoch}, Step: {now_step}, Train Loss These Steps: {train_step_loss}")
            print(f"Epoch: {epoch}, Step: {now_step}, Train Loss These Steps: {train_step_loss}")
            
            current_step_model_path = os.path.join(save_dir, f"model_step_{now_step}.pth.tar")
            save_model(model, optimizer, filename=current_step_model_path)
            logging.info(f"Model saved to {current_step_model_path}")
            print(f"Model saved to {current_step_model_path}")
            
            last_step_model_path = current_step_model_path
            
            train_step_loss = 0.0
             
        if (now_step + 1) % CFG.test_record_steps == 0:
            current_test_loss = validate_model(model, valid_loader, criterion, device)
            # 记录到TensorBoard
            #writer.add_scalar('Validation Loss', current_test_loss, now_step)
            # 保存当前的测试损失
            test_step_losses.append(current_test_loss)
            # 检查是否为最佳模型，并保存
            if current_test_loss < best_test_loss:
                best_test_loss = current_test_loss
                best_model_path = os.path.join(save_dir, f"model_step_best.pth.tar")
                
                # 删除上一次保存的最佳模型文件
                if last_best_model_path != best_model_path and os.path.exists(last_best_model_path):
                    os.remove(last_best_model_path)
                    logging.info(f"Deleted last best model at {last_best_model_path}")
                    

                save_model(model, optimizer, best_model_path)
                last_best_model_path = best_model_path
                print(f"New best model saved with validation loss: {current_test_loss}")
                
            print(f"Epoch: {epoch}, Step: {now_step}, Validation Loss: {current_test_loss}")

    scheduler.step()

#     train_loss /= len(train_loader)
#     train_losses.append(train_loss)

#     test_loss /= len(valid_loader)
#     test_losses.append(test_loss)
 

        
#     # 记录每个epoch的平均loss
#     writer.add_scalars('Epoch Losses', {'Training': train_loss, 'Validation': test_loss}, epoch)

#     logging.info(f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")
#     print(f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")

# 关闭writer
#writer.close()

'''
python predict_step_main.py > predict_step_main.log 2>&1
tail -f predict_step_main.log    
'''



