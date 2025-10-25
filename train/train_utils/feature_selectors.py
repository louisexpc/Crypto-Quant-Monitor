import lightgbm as lgb
from typing import List
import pandas as pd
class feature_selector:
    def __init__(self,df:pd.DataFrame,target:str):
        
        self.df = df
        self.target = target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")
        
        self.X = df.drop(columns=[target])
        self.y = df[target]
        self.selected_features = None
        self.lgb_dataset = lgb.Dataset(self.X, label=self.y)
    
    def train_model(self, params=None, num_boost_round=100, p_threshold=0.01):
        if params is None:
            params = {
                'objective': 'binary',
                'boosting_type': 'gbdt',
                'metric': 'binary_logloss',
                'verbosity': -1
            }
        self.model = lgb.train(params, self.lgb_dataset, num_boost_round=num_boost_round)
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()
        total_gain = importance.sum()
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'gain': importance
        })
        df_importance['relative_gain'] = df_importance['gain'] / total_gain
        

        # Select features based on importance
        self.selected_features  = df_importance.loc[
            df_importance['relative_gain'] >= p_threshold, 'feature'
        ].tolist()
        print(f"[INFO] Selected {len(self.selected_features)} features based on importance threshold {p_threshold}.")

    def get_selected_features(self) -> List[str]:
        if self.selected_features is None:
            raise ValueError("Model has not been trained yet.")
        return self.selected_features


    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        if self.selected_features is None:
            raise ValueError("Model has not been trained yet.")
        return df[self.selected_features]

