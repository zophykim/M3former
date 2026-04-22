import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.timefeatures import time_features
import warnings
import torch
warnings.filterwarnings('ignore')

FIXED_MIN = {
    "Longitude": 7.5,
    "Latitude": 53.5,
    "SOG": 0.0,
    "COG": 0.0
}

FIXED_MAX = {
    "Longitude": 14.0,
    "Latitude": 59.0,
    "SOG": 30.0,
    "COG": 360.0
}


class FixedMinMaxScaler:
    """
    只对指定维度做 min-max 归一化，其余维度保持不变
    """
    def __init__(self, min_dict, max_dict, feature_names):
        """
        min_dict / max_dict: {feature_name: value}
        feature_names: model_features 的顺序
        """
        self.feature_names = feature_names
        self.n_features_in_ = len(feature_names)

        self.min_ = np.zeros(self.n_features_in_)
        self.scale_ = np.ones(self.n_features_in_)

        for i, name in enumerate(feature_names):
            if name in min_dict:
                min_v = min_dict[name]
                max_v = max_dict[name]
                self.min_[i] = min_v
                self.scale_[i] = max_v - min_v
            else:
                # 不归一化的维度：x -> x
                self.min_[i] = 0.0
                self.scale_[i] = 1.0

    def transform(self, x):
        """
        x: (..., n_features)
        """
        return (x - self.min_) / self.scale_

    def inverse_transform(self, x):
        return x * self.scale_ + self.min_


def cyclical_encode_deg(x):
    """ x: degrees 0-360 """
    rad = np.deg2rad(x)
    return np.sin(rad), np.cos(rad)

class Dataset_AIS(Dataset):
    def __init__(self, csv_path, size=(24, 12, 12), 
                 features=['Longitude','Latitude','SOG','COG','Heading'],
                 static_features=['MMSI','Ship type','Width','Length','Draught','Destination','ETA'],
                 timeenc=1, freq='min',
                 scaler=None):
        """
        csv_path: CSV 文件路径
        size: tuple, (seq_len, label_len, pred_len)
        features: list, 滑动窗口数值特征
        static_features: list, 静态船舶特征
        timeenc: 0/1 时间特征编码方式
        freq: 时间步频率
        """
        super().__init__()
        self.seq_len, self.label_len, self.pred_len = size
        self.features = features
        self.static_features = static_features
        self.timeenc = timeenc
        self.freq = freq
        
        # ---- Load & sort ----
        self.df_raw = pd.read_csv(csv_path)
        self.df_raw = self.df_raw.sort_values(['MMSI','TimeStamp']).reset_index(drop=True)
        self.df_raw['TimeStamp'] = pd.to_numeric(self.df_raw['TimeStamp'], errors='coerce')
        self.df_raw['date'] = pd.to_datetime(self.df_raw['TimeStamp'], unit='s')

        # ---- Convert to numeric ----
        self.df_raw[self.features] = self.df_raw[self.features].apply(pd.to_numeric, errors='coerce')
        # self.df_raw = self.df_raw.dropna(subset=self.features)  # 去掉非数值行

        # ---- Cyclical transforms for COG / Heading ----
        COG_sin, COG_cos = cyclical_encode_deg(self.df_raw['COG'].values)
        HD_sin, HD_cos   = cyclical_encode_deg(self.df_raw['Heading'].values)

        self.df_raw['COG_sin'] = COG_sin
        self.df_raw['COG_cos'] = COG_cos
        self.df_raw['HD_sin']  = HD_sin
        self.df_raw['HD_cos']  = HD_cos

        self.model_features = [
            "Longitude", "Latitude", "SOG", "COG",
            "COG_sin", "COG_cos",
            "HD_sin", "HD_cos"
        ]
        self.scaler = FixedMinMaxScaler(
            min_dict=FIXED_MIN,
            max_dict=FIXED_MAX,
            feature_names=self.model_features
        )
   
        self.feature2idx = {f: i for i, f in enumerate(self.features)}

        self.raw_values = self.df_raw[self.features].values  # shape [N, F]
        self.raw_values = self.raw_values.astype(np.float32)

        self.data_values = self.scaler.transform(
            self.df_raw[self.model_features].values
        )
        self.data_values = self.data_values.astype(np.float32)


        self.mmsi_indices = self.df_raw.groupby('MMSI').indices

        # 静态特征映射: MMSI -> static 特征
        # self.mmsi_static_map = self.df_raw.groupby('MMSI')[self.static_features].first().to_dict('index')

        # 按 MMSI 生成滑动窗口索引
        self.indices = np.array(self._generate_indices(), dtype=np.int32)

        # 时间特征
        self.data_stamp = self._generate_time_features()
        self.interval = '10 min'
        self._generate_statsic_features()

    def _generate_statsic_features(self):
        self.mmsi_static_map = {}
        self.mmsi_stats_map = {}
        self.mmsi_desc_map = {} 
        for mmsi, idxs in self.mmsi_indices.items():
            static_info = self.df_raw.loc[idxs[0], self.static_features].to_dict()  # 静态特征
            self.mmsi_static_map[mmsi] = static_info
            
            # 预计算 max/min/mean，shape = (n_points, 3)
            vessel_data = self.raw_values[idxs, :]
            # 只取 SOG / Heading / COG 列索引
            sog = vessel_data[:, self.feature2idx["SOG"]]
            hdg = vessel_data[:, self.feature2idx["Heading"]]
            cog = vessel_data[:, self.feature2idx["COG"]]
            
            stats_array = np.stack([
                sog.max(axis=0), sog.min(axis=0), sog.mean(axis=0),
                hdg.max(axis=0), hdg.min(axis=0), hdg.mean(axis=0),
                cog.max(axis=0), cog.min(axis=0), cog.mean(axis=0)
            ])
            self.mmsi_stats_map[mmsi] = stats_array
            self.mmsi_desc_map[mmsi] = (
                f"A {static_info['Ship type'].lower()} vessel (MMSI {static_info['MMSI']}) "
                f"with length {static_info['Length']} m, width {static_info['Width']} m, "
                f"draught {static_info['Draught']} m. Destination \"{static_info['Destination']}\" "
                f"ETA {static_info['ETA']}"
            )

    def _generate_indices(self):
        """
        生成滑动窗口索引，保证窗口内同一个 MMSI
        """
        indices = []
        mmsi_groups = self.mmsi_indices
        for mmsi, idxs in mmsi_groups.items():
            n_points = len(idxs)
            max_start = n_points - self.seq_len - self.pred_len 
            for start in range(max_start + 1):
                indices.append((start, mmsi))
        return indices

    def _generate_time_features(self):
        """
        根据时间戳生成时间特征
        """
        if self.timeenc == 0:
            df_stamp = self.df_raw[['date']].copy()
            df_stamp['month'] = df_stamp.date.dt.month       # 月份 1-12
            df_stamp['day'] = df_stamp.date.dt.day           # 日 1-31
            df_stamp['weekday'] = df_stamp.date.dt.weekday   # 星期 0=周一
            df_stamp['hour'] = df_stamp.date.dt.hour         # 小时 0-23
            df_stamp['minute'] = df_stamp.date.dt.minute     # 分钟 0-59
            # 如果需要秒，也可以加：df_stamp['second'] = df_stamp.date.dt.second
            return df_stamp.drop(['date'], axis=1).values

        else:
            return time_features(pd.to_datetime(self.df_raw['date'].values), freq=self.freq).transpose(1,0)
    
    def __getitem__(self, index):
        start, mmsi = self.indices[index]
        idxs = self.mmsi_indices[mmsi]

        seq_x = self.data_values[idxs[start : start+self.seq_len]]
        seq_y = self.data_values[idxs[start+self.seq_len - self.label_len : start+self.seq_len+self.pred_len]]

        seq_x_mark = self.data_stamp[idxs[start : start+self.seq_len]]
        seq_y_mark = self.data_stamp[idxs[start+self.seq_len - self.label_len : start+self.seq_len+self.pred_len]]

        # ---- Static ----
        static_info = self.mmsi_static_map[mmsi]

        # ---- Stats use real angles ----
        seq_x_real = self.raw_values[idxs[start:start+self.seq_len]]


        sog = seq_x_real[:, self.feature2idx["SOG"]]
        cog = seq_x_real[:, self.feature2idx["COG"]]
        hdg = seq_x_real[:, self.feature2idx["Heading"]]
        stats_array = self.mmsi_stats_map[mmsi]
        stats = {
            "vessel desc": self.mmsi_desc_map[mmsi],
            "start_time": str(self.df_raw.loc[idxs[start], 'date']),
            "end_time": str(self.df_raw.loc[idxs[start+self.seq_len-1], 'date']),
            "interval": self.interval,
            "max_SOG": float(stats_array[0]),
            "min_SOG": float(stats_array[1]),
            "mean_SOG": float(stats_array[2]),
            "max_heading": float(stats_array[3]),
            "min_heading": float(stats_array[4]),
            "mean_heading": float(stats_array[5]),
            "max_COG": float(stats_array[6]),
            "min_COG": float(stats_array[7]),
            "mean_COG": float(stats_array[8]),
            "Destination": str(static_info['Destination']),
        }

        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
            torch.from_numpy(seq_x_mark).float(),
            torch.from_numpy(seq_y_mark).float(),
            static_info,
            stats
        )


    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
