import lightgbm as lgb
from typing import List
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from scipy.stats import rankdata, norm
from scipy.stats import mannwhitneyu
class LGB_feature_selector:
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

class BiserialRankEvaluator():
    name = "biserial_rank"

    def __init__(self, event_df: pd.DataFrame, df: pd.DataFrame, lookback: int = 36):
        """
        Args:
            - event_df: 包含事件資料的 DataFrame，即 label.py 所產出之格式，起碼包含 `t0` and `label`。
            - df: 原始資料的 DataFrame，包含所有特徵。
            - lookback: 每個事件的特徵長度（timepoints），即從 t0 往回算多少筆特徵(包括 t0)。
        """
        self.event_df = event_df
        self.df = df
        self.lookback = lookback


        self.event_df['t0'] = pd.to_datetime(self.event_df['t0'], errors='coerce', utc=True)
        
        self.df = self.df.reset_index()
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], errors='coerce', utc=True)

        self.event_df = self.event_df.sort_values('t0').reset_index(drop=True)
        self.df = self.df.sort_values('datetime').reset_index(drop=True)

        self.y_label = self.event_df['label'].to_numpy(dtype=int)
        self.event_df['event_id'] = np.arange(len(self.event_df)).astype(int)

        self.x_series_start_idx = {}
        for i, row in self.event_df.iterrows():
            self.x_series_start_idx[row['event_id']] = self.df[self.df['datetime'] == row['t0']].index[0]


    def evaluate(self, fitness_type: str = "fixed", threshold: float = 0.1) -> Tuple[pd.DataFrame, List[str]]:
        """
        評估 single individual 的 biserial rank correlation fitness。
        Args:
            - individual: 具有 tree.eval(df) 方法的個體。
            - fitness_type: "fixed" or "random"，決定使用固定效果或隨機效果的相關係數作為 fitness。
                - "fixed": 使用固定效果的 r_fixed。固定效應模型 (Fixed-effect model) 下的平均 rank-biserial correlation。
                - "random": 使用隨機效果的 r_random。隨機效應模型 (Random-effect model) 下的平均 rank-biserial correlation。
        Returns:
            - fitness: float
            - metrics: Dict[str, Any], 包含 'fixed_r' 與 'random_r'。
                - Rank-biserial correlation 的範圍為 [-1, 1]，fitness 取其絕對值。
        """
        pass_feature_names = []  # 保留 datetime 欄位
        # 建立特徵矩陣 X 與標籤 y
        # print(f"[Debug] Start evaluating individual with tree: {individual.show()}")
        for col in self.df.columns:
            if col == 'datetime':
                continue
            feature = self.df[col].to_numpy(dtype=float)
            n_events = len(self.event_df)
            m_timepoints = self.lookback

            # [MOD] 用 NaN 當 padding，避免人造 ties 導致 Var(U)=0→Var(r)=0→1/var_r 溢位
            X = np.full((n_events, m_timepoints), np.nan, dtype=float)
            y = self.y_label  # shape (n_events,)

            for i, row in self.event_df.iterrows():
                start_idx = self.x_series_start_idx[row['event_id']]
                # 擷取 lookback 長度的特徵: 包含 start_idx 本身
                start = (start_idx - self.lookback + 1) if (start_idx - self.lookback + 1) >= 0 else 0
                # 若前段不足 lookback，前面會留 NaN（保留，不以 0 補值）
                seg = feature[start: start_idx + 1]
                X[i, -len(seg):] = seg  # [MOD] 從尾端對齊，維持「含 t0 在末端」的時間語義

            # 計算 rank-biserial correlation（支援 NaN，僅以有值樣本計算）
            res = self.per_timepoint_rank_biserial(X, y)
            r_list = res['r_rb']           # shape (m,)
            var_list = res['var_r_rb']     # shape (m,)

            # [MOD] 在 meta-analysis 前過濾不可用 timepoints
            valid = np.isfinite(r_list) & np.isfinite(var_list) & (var_list > 0)
            if not np.any(valid):
                fixed = {'r_fixed': np.nan}
                random = {'r_random': np.nan}
                # 可選 debug
                # print(f"[Debug] usable timepoints: 0 / {len(r_list)}")
            else:
                # 可選 debug
                # print(f"[Debug] usable timepoints: {valid.sum()} / {len(r_list)}")
                fixed = self.fixed_effect_meta(r_list[valid], var_list[valid])
                random = self.random_effect_meta_dersimonian_laird(r_list[valid], var_list[valid])

            if fitness_type == "fixed":
                fitness = abs(fixed['r_fixed']) if not np.isnan(fixed['r_fixed']) else 0.0
            else:
                fitness = abs(random['r_random']) if not np.isnan(random['r_random']) else 0.0
            print(f"[INFO] Feature '{col}': fitness = {fitness:.6f} ({fitness_type})")
            if fitness > threshold:
                pass_feature_names.append(col)

        return self.df[pass_feature_names], pass_feature_names

    @staticmethod
    def per_timepoint_rank_biserial(X: np.ndarray, y: np.ndarray):
        """
        計算每個 timepoint 的 rank-biserial correlation 與相關統計值。

        Inputs:
            X: np.ndarray, shape (n, m), dtype float (或 int).
            每一 row = 一個事件 (sample)，每 column = 一個 timepoint / feature。
            y: np.ndarray, shape (n,), dtype int (0 or 1).
            事件的二元標籤 (0/1)。

        Returns (dict of np.ndarray, all length m):
            'r_rb'        : rank-biserial r (shape (m,), dtype=float)
            'U'           : Mann-Whitney U (for group y==1) (shape (m,))
            'p_two_sided' : two-sided p-value from Mann-Whitney (shape (m,))
            'mean_rank_1' : mean rank of group y==1 (shape (m,))
            'n1'          : count of y==1 (scalar int)
            'n0'          : count of y==0 (scalar int)
            'var_r_rb'    : approximate variance of r_rb with tie correction (shape (m,))
            'z'           : z-score for r_rb / pooled test (shape (m,))
            'ci_lower'    : 95% CI lower bound for r_rb (shape (m,))
            'ci_upper'    : 95% CI upper bound for r_rb (shape (m,))
        """
        if X.ndim != 2:
            raise ValueError("X must be 2-D array shape (n, m)")
        n, m = X.shape
        if y.shape != (n,):
            raise ValueError("y must have shape (n,)")

        # ensure y is 0/1 integers
        y = y.astype(int)
        n1_global = int((y == 1).sum())
        n0_global = int((y == 0).sum())
        N_global = n1_global + n0_global
        if n1_global == 0 or n0_global == 0:
            raise ValueError("Both classes 0 and 1 must be present.")

        r_rb = np.empty(m, dtype=float)
        U_arr = np.empty(m, dtype=float)
        p_arr = np.empty(m, dtype=float)
        mean_rank_1 = np.empty(m, dtype=float)
        var_r = np.empty(m, dtype=float)
        z_arr = np.empty(m, dtype=float)
        ci_low = np.empty(m, dtype=float)
        ci_up = np.empty(m, dtype=float)

        for j in range(m):
            col = X[:, j]

            # [MOD] 僅使用有效（非 NaN）樣本
            mask = np.isfinite(col)
            col_j = col[mask]
            y_j = y[mask]

            n1_j = int((y_j == 1).sum())
            n0_j = int((y_j == 0).sum())
            N_j = n1_j + n0_j

            # [MOD] 樣本不足或單邊類別 → 回傳 NaN，留待上游濾除
            if N_j < 2 or n1_j == 0 or n0_j == 0:
                r_rb[j] = np.nan
                U_arr[j] = np.nan
                p_arr[j] = np.nan
                mean_rank_1[j] = np.nan
                var_r[j] = np.nan
                z_arr[j] = 0.0
                ci_low[j] = np.nan
                ci_up[j] = np.nan
                continue

            # [MOD] 欄位全常數（或全部值相等） → 無秩差，Var(U)=0，直接回 NaN
            if np.nanmax(col_j) == np.nanmin(col_j):
                r_rb[j] = np.nan
                U_arr[j] = np.nan
                p_arr[j] = np.nan
                mean_rank_1[j] = np.nan
                var_r[j] = np.nan
                z_arr[j] = 0.0
                ci_low[j] = np.nan
                ci_up[j] = np.nan
                continue

            # ranks (average ranks for ties) 僅就有效樣本計算
            ranks_j = rankdata(col_j, method='average')
            R1 = np.sum(ranks_j[y_j == 1])
            mean_rank_1[j] = R1 / n1_j

            # U for group1
            U = R1 - n1_j * (n1_j + 1) / 2.0
            U_arr[j] = U

            # Mann-Whitney p-value（scipy 對 ties 會走常態近似）
            grp0 = col_j[y_j == 0]
            grp1 = col_j[y_j == 1]
            try:
                res = mannwhitneyu(grp1, grp0, alternative='two-sided', method='asymptotic')
                p_val = res.pvalue
            except TypeError:
                # older scipy versions do not have method=; fallback
                res = mannwhitneyu(grp1, grp0, alternative='two-sided')
                p_val = res.pvalue
            p_arr[j] = p_val

            # compute r_rb
            r = (2.0 * U) / (n1_j * n0_j) - 1.0
            r_rb[j] = r

            # tie correction for Var(U):
            uniq_vals, counts = np.unique(col_j, return_counts=True)
            ties = counts[counts > 1]
            if ties.size == 0:
                S = 0.0
            else:
                S = np.sum(ties.astype(float) ** 3 - ties.astype(float))

            # Var(U) with tie correction:
            # Var(U) = n1*n0/12 * (N+1 - S/(N*(N-1)))
            if N_j * (N_j - 1) == 0:
                var_U = 0.0
            else:
                var_U = (n1_j * n0_j / 12.0) * (N_j + 1.0 - S / (N_j * (N_j - 1.0)))

            # Var(r) by delta method: r = a * U + b, where a = 2/(n1*n0)
            a = 2.0 / (n1_j * n0_j)
            # [MOD] 若 var_U 為 0，改回 NaN（讓上游 meta 濾掉）
            var_r[j] = (a ** 2) * var_U if var_U > 0 else np.nan

            # z and CI for r (normal approx)
            se_r = np.sqrt(var_r[j]) if np.isfinite(var_r[j]) and var_r[j] > 0 else 0.0
            if se_r > 0:
                z_arr[j] = r / se_r
                ci_low[j] = r - norm.ppf(0.975) * se_r
                ci_up[j] = r + norm.ppf(0.975) * se_r
            else:
                z_arr[j] = 0.0
                ci_low[j] = r
                ci_up[j] = r

        # 回傳時的 n1/n0 保留全域統計（不影響上游 meta）
        return {
            'r_rb': r_rb,
            'U': U_arr,
            'p_two_sided': p_arr,
            'mean_rank_1': mean_rank_1,
            'n1': n1_global,
            'n0': n0_global,
            'var_r_rb': var_r,
            'z': z_arr,
            'ci_lower': ci_low,
            'ci_upper': ci_up
        }

    @staticmethod
    def fixed_effect_meta(r: np.ndarray, var_r: np.ndarray):
        """
        固定效果的加權平均 (inverse-variance weighting)。

        Inputs:
            r: np.ndarray shape (k,)   = 每個效應量 r_i
            var_r: np.ndarray shape (k,) = 每個效應量的 variance

        Returns dict:
            'r_fixed' : pooled r
            'se'      : standard error of pooled r
            'z'       : z-score
            'p'       : two-sided p-value
            'ci_lower','ci_upper'
        """
        if r.shape != var_r.shape:
            raise ValueError("r and var_r must have same shape")

        # [MOD] 安全過濾：只保留有限且 var>0 的點
        mask = np.isfinite(r) & np.isfinite(var_r) & (var_r > 0)
        if not np.any(mask):
            return {'r_fixed': np.nan, 'se': np.nan, 'z': 0.0, 'p': 1.0,
                    'ci_lower': np.nan, 'ci_upper': np.nan}

        r = r[mask]
        var_r = var_r[mask]

        w = 1.0 / var_r
        sumw = np.sum(w)
        r_fixed = np.sum(w * r) / sumw
        se = np.sqrt(1.0 / sumw)
        z = r_fixed / se if se > 0 else 0.0
        p = 2.0 * (1.0 - norm.cdf(abs(z)))
        ci_l = r_fixed - norm.ppf(0.975) * se
        ci_u = r_fixed + norm.ppf(0.975) * se
        return {'r_fixed': r_fixed, 'se': se, 'z': z, 'p': p, 'ci_lower': ci_l, 'ci_upper': ci_u}

    @staticmethod
    def random_effect_meta_dersimonian_laird(r: np.ndarray, var_r: np.ndarray):
        """
        DerSimonian-Laird 隨機效果合併，回傳 pooled effect 及 heterogeneity 指標 I^2、tau^2 等。

        Inputs:
            r: np.ndarray shape (k,)
            var_r: np.ndarray shape (k,)

        Returns dict:
            'r_random','se','z','p','ci_lower','ci_upper',
            'tau2' (between-study variance), 'Q', 'I2' (percent)
        """
        # [MOD] 安全過濾：只保留有限且 var>0 的點
        mask = np.isfinite(r) & np.isfinite(var_r) & (var_r > 0)
        if not np.any(mask):
            return {
                'r_random': np.nan, 'se': np.nan, 'z': 0.0, 'p': 1.0,
                'ci_lower': np.nan, 'ci_upper': np.nan,
                'tau2': np.nan, 'Q': np.nan, 'I2': np.nan
            }

        r = r[mask]
        var_r = var_r[mask]
        k = len(r)

        w = 1.0 / var_r
        sumw = np.sum(w)
        r_fixed = np.sum(w * r) / sumw
        Q = np.sum(w * (r - r_fixed) ** 2)

        C = sumw - np.sum(w ** 2) / sumw
        if C <= 0:
            tau2 = 0.0
        else:
            tau2 = max(0.0, (Q - (k - 1)) / C)

        w_star = 1.0 / (var_r + tau2)
        sumw_star = np.sum(w_star)
        r_random = np.sum(w_star * r) / sumw_star
        se = np.sqrt(1.0 / sumw_star)
        z = r_random / se if se > 0 else 0.0
        p = 2.0 * (1.0 - norm.cdf(abs(z)))
        ci_l = r_random - norm.ppf(0.975) * se
        ci_u = r_random + norm.ppf(0.975) * se
        I2 = max(0.0, (Q - (k - 1)) / Q) * 100.0 if Q > (k - 1) else 0.0

        return {
            'r_random': r_random, 'se': se, 'z': z, 'p': p, 'ci_lower': ci_l, 'ci_upper': ci_u,
            'tau2': tau2, 'Q': Q, 'I2': I2
        }