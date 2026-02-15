# DL Module
## config
- `configs/config.yaml` : 主要參數管理，有需要的話按照格式新增
## Feature
- `utils/feature_computer.py`:
    ```python
    import pandas as pd
    class Feature_Computer():
        def __init__(self, config:dict):
            """
            此物件長存於記憶體中，負責計算技術指標。透過 YAML 配置檔初始化(configs/config.yaml)。
            Args:
                - config (dict): 配置字典，包含技術指標計算參數
                - another args if needed
            Hint:
            - config 可能包含的參數:
                - long_feature_cfg_path: str, 長期技術指標配置檔路徑
                - short_feature_cfg_path: str, 短期技術指標配置檔路徑
                ...etc
            - 原則上多空頭分開兩個物件維護
            
            """
            pass
        def compute(self, df_raw:pd.DataFrame)->pd.DataFrame:
            """
            接收一個原始 OHLCV + FNG Index 的 DataFrame，計算並回傳包含技術指標的 DataFrame
            需要說明df raw:
            - Time frame(與模型輸入對齊)

            Args:
                - df_raw (pd.DataFrame): 原始 OHLCV + FNG Index DataFrame
                - another args if needed
            Returns:
                - pd.DataFrame: 包含技術指標的 DataFrame (可以直接餵入模型)
            """
            #TODO
        # Other methods if needed
    ```
## Model
- `utils/predictor.py`
    ```python
    class Predictor():
        def __init__(self, config:str):
            """
            此物件長存於記憶體中，負責載入並執行預測模型。透過 YAML 配置檔初始化(configs/config.yaml)。
            Args:
                - config (dict): 配置字典，包含技術指標計算參數
                - another args if needed
            Hint:
            - config 可能包含的參數:
                - model_path: str, 預測模型檔案路徑
                - model_args_path: str, 模型其他參數設定(如果有，單獨整理成一個 yaml 檔案)
                ...etc
            - 原則上多空頭分開兩個物件維護，所以config 會有 long / short 之分
            """
            pass

        def predict(self, features:pd.DataFrame)->Tuple[float, bool]:
            """
            接收一個包含技術指標的 DataFrame，執行預測(單一模型)並回傳預測結果 DataFrame
            Args:
                - features (pd.DataFrame): 包含技術指標的 DataFrame
                - another args if needed
            Returns:
                - float: Model inference 消耗時間 (秒)，提供回測滑點參考
                - bool: 預測標籤 (True: 進場，False: 不進場)
            """
            #TODO
        def predict_vote(self, features:pd.DataFrame)->Tuple[float, bool]:
            """
            接收一個包含技術指標的 DataFrame，執行多模型投票預測並回傳預測結果 DataFrame
            Args:
                - features (pd.DataFrame): 包含技術指標的 DataFrame
                - another args if needed
            Returns:
                - float: Model inference 消耗時間 (秒)，提供回測滑點參考
                - bool: 預測標籤 (True: 進場，False: 不進場)
            """
            #TODO

    ```
