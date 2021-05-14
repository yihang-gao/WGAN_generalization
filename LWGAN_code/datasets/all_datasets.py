import pandas as pd
from pathlib import Path
from utils import base_dataset
from utils import registry


@registry.register_dataset(name='MovieLen')
class MovieLen(base_dataset.BaseDataset):

    def __init__(self,
                 path='datasets/ml-25m/',
                 filename='ratings.csv',
                 min_length=5,
                 max_length=50,
                 positive_score=1,
                 num_negative_sample=1,
                 min_interval=180,
                 num_interval=64):
        super().__init__(path,
                         filename,
                         min_length,
                         max_length,
                         positive_score,
                         num_negative_sample,
                         min_interval,
                         num_interval)

    def load_raw_data(self):
        data = pd.read_csv(str(Path(self._path, self._filename)), sep='::', engine='python',
                           names=['user_id', 'item_id', 'score', 'timestamp'])
        # data = pd.read_csv(str(Path(self._path, self._filename)), sep=',', engine='python')
        # data.columns = ['user_id', 'item_id', 'score', 'timestamp']

        # Only treat item with at least `positive_score` as positive samples
        data = data[data['score'] >= self._positive_score]

        # Filtering out items with low frequency and users with short history
        data['item_count'] = data.groupby('item_id')['item_id'].transform('count')
        data['user_count'] = data.groupby('user_id')['user_id'].transform('count')
        data = data[(data['item_count'] >= 5) & (data['user_count'] >= self._min_length)]
        # Sort by user id, timestamp
        data = data.sort_values(by=['user_id', 'timestamp'])
        return data
