# 基於遺傳演算法的隨機Alpha生成器
# Ref : https://arxiv.org/html/2412.00896v1


import hashlib
import pandas as pd
import numpy as np
import random
import copy
import os
from typing import Any, List, Optional, Tuple
from node import Node, Leaf, OpNode
import json, time
import talib
from pprint import pprint,pformat
from evaluation import BaseEvaluator, DefaultEvaluator, BiserialRankEvaluator
from ind import Individual
# ========== multiprocessing workers (NEW) ==========
from multiprocessing import get_context

_G_DF = None
_G_RET = None
_G_FTYPE = None
_G_EVAL = None
def _mp_init(df, returns, fitness_type, evaluator):
    """子程序初始化：把大物件放到全域，避免每次 pickling 傳遞。"""
    global _G_DF, _G_RET, _G_FTYPE, _G_EVAL
    _G_DF, _G_RET, _G_FTYPE, _G_EVAL = df, returns, fitness_type, evaluator

def _mp_eval_worker(args):
    idx, ind = args
    try:
        # 確保 individual 有 evaluator（指向全域 _G_EVAL）

        ind.evaluator = _G_EVAL
        fit = ind.evaluate(_G_DF, _G_RET, _G_FTYPE)
        # evaluate 已把 ic/sharpe 寫回個體
        return idx, fit, ind.ic, ind.sharpe, ind.fixed_r, ind.random_r
    except Exception:
        return idx, 0.0, np.nan, np.nan, np.nan, np.nan


# ================== 樹構造函數 ==================

def random_tree(depth=3, terminals=['open', 'close', 'high', 'low', 'volume'], 
                operators=['+', '-', '*', '/', 'rolling_mean']):
    """隨機生成alpha公式樹"""
    if depth == 0:
        if random.random() < 0.7:
            return Leaf(random.choice(terminals))
        else:
            return Leaf(random.uniform(1, 10))
    else:
        operator = random.choice(operators)
        left = random_tree(depth-1, terminals, operators)
        
        if operator in ['rolling_mean', 'rolling_std']:
            right = Leaf(random.randint(2, 10))
        else:
            right = random_tree(depth-1, terminals, operators)
        return OpNode(operator, left, right)

# ================== 評估函數 ==================
def safe_corr(x, y, use_rank: bool =False, min_n:int=10, eps:float=1e-12)-> float:
    """
    安全版相關係數計算: For 單一股票池時間序列與時間序列回報的相關性計算
    若 use_rank=True，則計算 Rank Correlation（Spearman），若 False 則計算 Pearson
    Args:
    - x, y: 1D array-like
    - use_rank: 是否使用秩相關
    - min_n: 最小有效樣本數
    - eps: 標準差下限，避免除以零
    Returns:
    - correlation coefficient or np.nan if invalid   
    """
    import numpy as np, pandas as pd
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < min_n: return np.nan
    if use_rank:
        rx = pd.Series(x).rank(pct=False).to_numpy()
        ry = pd.Series(y).rank(pct=False).to_numpy()
        x, y = rx, ry
    xm, ym = x.mean(), y.mean()
    xc, yc = x - xm, y - ym
    sx = np.sqrt((xc * xc).mean()); sy = np.sqrt((yc * yc).mean())
    if not np.isfinite(sx) or not np.isfinite(sy) or sx < eps or sy < eps:
        return np.nan
    cov = (xc * yc).mean()
    return cov / (sx * sy)

def daily_rank_ic(df, date_col, alpha_col, ret_col, min_n=20):
    """ 計算每日的 Rank IC: 目前無法使用 """
    out, idx = [], []
    for d, g in df[[date_col, alpha_col, ret_col]].dropna().groupby(date_col):
        a = g[alpha_col].to_numpy(); r = g[ret_col].to_numpy()
        if len(a) < min_n or pd.Series(a).nunique() < 2 or pd.Series(r).nunique() < 2:
            out.append(np.nan)
        else:
            out.append(safe_corr(a, r, use_rank=True, min_n=min_n))
        idx.append(d)
    return pd.Series(out, index=idx).sort_index()

def safe_sharpe(strategy_returns, ann:int = 252, eps: float=1e-12)->float:
    """
    安全版 Sharpe 計算
    Args:
    - strategy_returns: 策略日收益率序列
    - ann: 年化倍數（252交易日）
    - eps: 標準差下限，避免除以零
    Returns:
    - Sharpe ratio or np.nan if invalid
    """
    r = pd.Series(strategy_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.size < 2: return np.nan
    mu = r.mean(); sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd < eps: return np.nan
    return mu / sd * np.sqrt(ann)


# =========　API　=========


def calc_ic(signal, returns, use_rank=False, min_n=10):
    """時間序列 IC（單標的）；若改做截面 RankIC，見下方備註。"""
    if signal is None: return np.nan
    s = pd.concat([signal, returns], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < min_n: return np.nan
    return safe_corr(s.iloc[:,0].values, s.iloc[:,1].values, use_rank=use_rank, min_n=min_n)

def calc_sharpe(signal, returns)-> float:
    """信號多空策略的 Sharpe（安全版）
    備註：此為單標的時間序列 Sharpe，若要做截面 Sharpe，需先計算每日多空組合收益率序列，再計算其 Sharpe。

    Args:
    - signal: 時間序列信號（可為 pd.Series 或 np.ndarray）
    - returns: 時間序列回報（可為 pd.Series 或 np.ndarray）
    Returns:
    - Sharpe ratio or np.nan if invalid
    """
    if signal is None: return np.nan
    positions = np.sign(pd.Series(signal).fillna(0))
    strategy_returns = (positions.shift(1) * returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(strategy_returns) == 0: return np.nan
    return safe_sharpe(strategy_returns)

# ================== 儲存工具 =======================
# --- 序列化 ---
def node_to_dict(node):
    if isinstance(node, Leaf):
        return {"type": "Leaf", "value": node.value}
    elif isinstance(node, OpNode):
        d = {
            "type": "OpNode",
            "operator": node.operator,
            "arity": getattr(node, "arity", None),
            "left": node_to_dict(node.left) if getattr(node, "left", None) is not None else None,
            "right": node_to_dict(node.right) if getattr(node, "right", None) is not None else None,
        }
        # 若你真的有三元子樹（例如 correlation 的 window），一併存（可選）
        if hasattr(node, "third") and node.third is not None:
            d["third"] = node_to_dict(node.third)
        return d
    else:
        raise TypeError(f"Unknown node type: {type(node)}")

# --- 反序列化 ---
def node_from_dict(d):
    t = d["type"]
    if t == "Leaf":
        return Leaf(d["value"])
    elif t == "OpNode":
        left  = node_from_dict(d["left"])  if d.get("left")  is not None else None
        right = node_from_dict(d["right"]) if d.get("right") is not None else None
        opn = OpNode(d["operator"], left, right)
        # 可選：若你支援三元，還原 third
        if "third" in d and d["third"] is not None:
            opn.third = node_from_dict(d["third"])
        return opn
    else:
        raise ValueError(f"Unknown dict type: {t}")


        
def save_alpha(ind: Individual, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ind.to_dict(), f, ensure_ascii=False, indent=2)

def load_alpha(path: str) -> Individual:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return Individual.from_dict(d)

class GeneticAlphaSolver:
    """遺傳演算法主體"""
    def __init__(
            self, 
            df:pd.DataFrame, 
            returns:pd.Series, 
            generations:int=10, 
            fitness_type:str='ic', #Default: 'ic' / 'sharpe' , Biserial : 'fixed' / 'random'
            evaluator: Optional[BaseEvaluator] = None,
            point_mutation_rate:float=0.3, 
            crossover_rate:float=0.7,
            population_size:int =50, 
            tournament_size:int=5,
            depth:int=3,
            population = None | list[Individual],
            operator_set : list[str] = ["+", "-", "*", "/", "sqrt", "log", "inverse", "sigmoid",
                "rank", "scale", "signedpower", "delay", "covariance", "correlation", "delta",
                "decay_linear"],
            terminal_set : list[str] = ["open", "close", "high", "low", "volume"],
            early_stopping_generations:int = 20,
            # NEW multiprocessing
            n_jobs:int = 1,                 # NEW: 并行程序數 (1 = 關閉並行)
            mp_start_method:str | None = None  # NEW: 'spawn' / 'fork' / None→自動            
        ):
        """
        Args:
        - df: 特徵資料集（含時間序列特徵）
        - returns: 目標回報序列（與 df 對齊）
        - generations: 演化世代數
        - fitness_type: 適應度類型 ('ic', 'sharpe')
        - point_mutation_rate: 點突變率
        - crossover_rate: 交叉率
        - population_size: 種群大小
        - tournament_size: 錦標賽選擇大小
        - depth: 樹最大深度
        - population: 初始種群（None 則隨機生成一個簡單個體）
        - operator_set: 運算符集合
        - terminal_set: 終端符集合
        - early_stopping_generations: 若連續 N 代最佳適應度無提升則提前停止
        - n_jobs: NEW 并行程序數 (1 = 關閉并行)
        - mp_start_method: NEW 'spawn' / 'fork' / None→自動
        
        """
        self.df = df
        self.returns = returns
        self.population_size = population_size
        self.generations = generations
        self.fitness_type = fitness_type
        self.evaluator = evaluator or DefaultEvaluator()
        self.point_mutation_rate = point_mutation_rate
        self.best_history = []
        self.operator_set = operator_set
        self.terminal_set = terminal_set
        self.depth = depth
        self.tournament_size = tournament_size

        self.crossover_rate = crossover_rate
        self.early_stopping_generations = early_stopping_generations

        """
        根據論文描述，初始種群應該是簡單結構的 alpha 公式，若沒有指定，則隨機生成一個當作固定結構 alpha
        """
        if population is None:
            self.population = [Individual(random_tree())]
        else:
            self.population = population

        # NEW multiprocessing
        self.n_jobs = n_jobs
        self.mp_start_method = mp_start_method  # None → get_context() 內部選擇        



    def restrictedCrossover(self, parent1:Individual, parent2:Individual)-> Tuple[Individual, Individual] | None:
        # 1) 檢查整體結構相容（呼叫你已有的 _identical_structure）
        if not self._identical_structure(parent1.tree, parent2.tree):
            return None

        # 2) deep copy 以避免 alias
        treeA = copy.deepcopy(parent1.tree)
        treeB = copy.deepcopy(parent2.tree)
        # 記得設 parent pointers（deepcopy 可能保留 None，依實作需確保）
        self._set_parent_recursive(treeA, None)
        self._set_parent_recursive(treeB, None)

        child1 = Individual(treeA)
        child2 = Individual(treeB)

        # 3) 以相同 traversal 取得 pair list（同位置對應）
        nodesA = self._get_nodes_with_paths(child1.tree)
        nodesB = self._get_nodes_with_paths(child2.tree)

        if not nodesA or not nodesB or len(nodesA) != len(nodesB):
            # 若數量不符，退回原本（雖然 _identical_structure 應確保相等）
            return None

        # 4) 隨機挑一個 index（即同一位置）
        idx = random.randrange(len(nodesA))
        nodeA, pathA = nodesA[idx]
        nodeB, pathB = nodesB[idx]

        # 5) 交換：用 deepcopy 的子樹插入，避免共享 reference
        subtreeA_copy = copy.deepcopy(nodeA)
        subtreeB_copy = copy.deepcopy(nodeB)
        self._set_parent_recursive(subtreeA_copy, None)
        self._set_parent_recursive(subtreeB_copy, None)

        # replace in child1, child2
        new_tree1 = self._replace_subtree_by_path(child1.tree, pathA, subtreeB_copy)
        new_tree2 = self._replace_subtree_by_path(child2.tree, pathB, subtreeA_copy)

        # 6) 驗證 depth 或其他 global constraint
        if new_tree1.treeDepth() > self.depth or new_tree2.treeDepth() > self.depth:
            # 拒絕此次 crossover（或做 retry 機制）
            return None 

        # 7) 如果一切通過，回傳新的 Individual
        child1.tree = new_tree1
        child2.tree = new_tree2
        return child1, child2
    
    def _get_nodes_with_paths(self, root:Node) -> List[Tuple[Node, List[int]]]:
        """
        回傳 list of (node, path)；path 為 index list 表示 child 路徑，
        以 Binary (0:left, 1:right)表示路徑
        例如 [] = root, [0] = left, [1] = right, [0,1] = root.left.right
        """
        nodes = []
        def _rec(node:Node, path:List[int]):
            if node is None:
                return
            nodes.append((node, list(path)))
            # binary 假設
            if hasattr(node, "left") and node.left is not None:
                path.append(0); _rec(node.left, path); path.pop()
            if hasattr(node, "right") and node.right is not None:
                path.append(1); _rec(node.right, path); path.pop()
        _rec(root, [])
        return nodes
    def _get_node_by_path(self, root, path):
        node = root
        for idx in path:
            if idx == 0:
                node = node.left
            else:
                node = node.right
            if node is None:
                return None
        return node
    def _replace_subtree_by_path(self, root: Node, path: List[int], new_subtree: Node) -> Node:
        """
        以 path 指定位置替換子樹。回傳新的 root（注意：若替換 root，回傳 new_subtree）。
        並且設定 new_subtree.parent。
        這個函式會直接在原 tree 修改（in-place）。
        """
        if len(path) == 0:
            # replace root
            new_subtree.parent = None
            return new_subtree
        parent_path = path[:-1]
        parent = self._get_node_by_path(root, parent_path)
        if parent is None:
            raise RuntimeError("invalid parent path")
        idx = path[-1]
        if idx == 0:
            parent.left = new_subtree
        else:
            parent.right = new_subtree
        new_subtree.parent = parent
        return root
    def _set_parent_recursive(self, node, parent=None):
        """遞迴設定 parent pointer（new_subtree 是 deepcopy 後的新物件）"""
        if node is None:
            return
        node.parent = parent
        if hasattr(node, "left"):
            self._set_parent_recursive(node.left, node)
        if hasattr(node, "right"):
            self._set_parent_recursive(node.right, node)
    def _identical_structure_update(self, ind1:Individual, ind2:Individual)-> bool:
        """Update: 改為 dict 比較"""
        dict1 = ind1.node_to_canonical_dict(ind1.tree)
        dict2 = ind2.node_to_canonical_dict(ind2.tree)
        return self._compare_dicts(dict1, dict2)

    def _compare_dicts(self,d1, d2, ignore_keys=None):
        if ignore_keys is None:
            ignore_keys = {"value","op"}  # 預設忽略 value 欄位

        if isinstance(d1, dict) and isinstance(d2, dict):
            # 先比 key 集合，排除要忽略的 key
            keys1 = set(d1.keys()) - ignore_keys
            keys2 = set(d2.keys()) - ignore_keys
            if keys1 != keys2:
                return False
            # 遞迴檢查每個 key
            for k in keys1:
                if not self._compare_dicts(d1[k], d2[k], ignore_keys):
                    return False
            return True
        elif isinstance(d1, list) and isinstance(d2, list):
            if len(d1) != len(d2):
                return False
            return all(self._compare_dicts(x, y, ignore_keys) for x, y in zip(d1, d2))
        else:
            # leaf node -> 只有在不是忽略欄位時才要比對值
            return d1 == d2



    def _identical_structure(self, node1:Node, node2:Node, mode :str = 'relaxed')-> bool:
        """檢查兩個子樹是否結構相同: 
        簡化版，僅適用當前的運算符集(不超過二元的運算符)，未來拓展需要考量更多情況：
        - 葉節點（Leaf）要比較 terminal_role，而非單純回傳 True
        - child_roles 不應該只比對等於，而是做「相容性」判斷
        - 不要硬綁 binary（left/right）
        - 加入 strict/relaxed 模式
        - 交叉前後應有後驗驗證（validator）
        - 避免只比對 child_roles 的 list 等於

        Args:
            - node1 (Node): 樹1的根節點
            - node2 (Node): 樹2的根節點
            - mode (str): 'strict' or 'relaxed' 模式
        """
        if mode not in ['strict', 'relaxed']:
            raise ValueError("mode must be 'strict' or 'relaxed'")
        
        if type(node1) != type(node2):
            return False
        if isinstance(node1, Leaf):
            if  type(node1.value) == type(node2.value):
                if mode == 'strict':
                    return node1.value == node2.value
                elif mode == 'relaxed':
                    return True
            else:
                return False
            
        if isinstance(node1, OpNode):

            if mode == 'strict':
                if node1.operator != node2.operator:
                    return False

            if node1.arity!= node2.arity:
                return False
            
            if node1.node_class != node2.node_class:
                return False
            
            if node1.child_roles != node2.child_roles:
                return False
                
            return self._identical_structure(node1.left, node2.left) and \
                   self._identical_structure(node1.right, node2.right)
        return False

    def pointMutation(self, individual:Individual):
        """點突變：隨機改變樹中的一個節點"""
        ind_copy = copy.deepcopy(individual)
        self._pointMutation(ind_copy.tree)
        return ind_copy

    def _pointMutation(self, node:Node):
        """遞迴實現點突變: 單點變異版本"""
        if node is None:
            return
        
        if isinstance(node, Leaf):
            if random.random() < self.point_mutation_rate:  # 根據突變率決定是否突變
                if isinstance(node.value, str):
                    node.value = random.choice(self.terminal_set)
                else:
                    node.value = random.uniform(1, 10)
                return # 突變後不繼續遞迴
        elif isinstance(node, OpNode):
            if random.random() < self.point_mutation_rate:  # 根據突變率決定是否突變
                node.operator = random.choice(self.operator_set)
                return # 突變後不繼續遞迴
            self._pointMutation(node.left)
            if node.right:
                self._pointMutation(node.right)

        # ========== population evaluation (NEW) ==========
    def _evaluate_population(self, pop: list[Individual]):
        """
        以最小入侵方式把評估並行化：
        - n_jobs == 1: 完全沿用原本行為（逐一 ind.evaluate(...)）
        - n_jobs > 1 : multiprocess Pool（純計算，結果回寫）
        """
        if self.n_jobs <= 1:
            # --- 原本同步路徑（行為不變） ---
            for ind in pop:
                ind.evaluator = self.evaluator
                ind.evaluate(self.df, self.returns, self.fitness_type)
            return

        # --- 多進程路徑 ---
        ctx = get_context(self.mp_start_method or "spawn")  # 跨平臺穩定
        with ctx.Pool(processes=self.n_jobs,
                      initializer=_mp_init,
                      initargs=(self.df, self.returns, self.fitness_type, self.evaluator)) as pool:
            tasks = [(i, ind) for i, ind in enumerate(pop)]
            # 用 imap_unordered 加速回傳；以 idx 回寫，不影響排序穩定性
            chunksize = max(1, len(tasks) // (self.n_jobs * 4))
            for idx, fit, ic, sharpe, fixed_r, random_r in pool.imap_unordered(_mp_eval_worker, tasks, chunksize=chunksize):
                pop[idx].fitness = fit
                pop[idx].ic = ic
                pop[idx].sharpe = sharpe
                # 可選：若要記錄更多指標，可在 Individual 加欄位
                pop[idx].fixed_r = fixed_r
                pop[idx].random_r = random_r

                

    def selection(self, parents: List[Individual]=None) -> Individual:
        """錦標賽選擇"""
        if len(parents) <= self.tournament_size:
            size = len(parents)
        else:
            size = self.tournament_size

        tournament = random.sample(parents, size)
        winner = max(tournament, key=lambda x: x.fitness if x.fitness else 0)
        return copy.deepcopy(winner)

    def evolve(self):
        """主進化循環 : 實務上這裡應該平行化
        Algorithm 1: Warm Start GP Framework
        Input:
            F(x)        : Fitness function
            n_pop       : Population size
            p_crossover : Crossover probability
            p_mutation  : Mutation probability
            P           : Other GP params
            X           : Stock data
            Y           : Forward return
            alpha_init  : Initial alpha (warm start)

        Output:
            Best evolved alpha

        ------------------------------------------------------

        t ← 0
        Pop(0) ← { alpha_init }            # 初始族群只有一個給定的 alpha
        EvaluatePopulation(Pop(0))         # 計算適應度

        while termination condition NOT met do
            # Step 1: 保留前一代的最佳個體 (elitism)
            Pop(t+1) ← { best individual from Pop(t) }

            while |Pop(t+1)| < n_pop do
                # Step 2: 決定使用哪種操作
                if (t = 0 or t = 1) then
                    Mutation ← PointMutation          # 第一代只能做突變
                else
                    Mutation ← randomly choose {Crossover, PointMutation}
                end if

                # Step 3: 依據操作產生 offspring
                if Mutation = Crossover then
                    P1 ← TournamentSelection(Pop(t))
                    P2 ← TournamentSelection(Pop(t))
                    Offspring ← RestrictedCrossover(P1, P2)

                else if Mutation = PointMutation then
                    P1 ← TournamentSelection(Pop(t))
                    Offspring ← PointMutation(P1)

                else
                    # 如果沒有操作，就直接複製
                    P1 ← TournamentSelection(Pop(t))
                    Offspring ← P1
                end if

                # Step 4: 避免重複個體
                if Offspring NOT IN Pop(t+1) then
                    Insert Offspring into Pop(t+1)
                end if
            end while

            # Step 5: 計算新族群的適應度
            EvaluatePopulation(Pop(t+1))

            # Step 6: 迭代下一代
            t ← t + 1
        end while

        Return best individual from Pop(t)

        """
        # for ind in self.population:
        #     if ind.fitness is None:
        #         ind.evaluate(self.df, self.returns, self.fitness_type)
        self._evaluate_population(self.population)

        # Eealy Stop 機制
        best_fitness = max(ind.fitness if ind.fitness else 0 for ind in self.population)
        tolerance_trials = self.early_stopping_generations
        no_improvement_count = 0

        for gen in range(self.generations) :
            start_time = time.time()
            # Step 1: 保留前一代的最佳個體 (elitism)
            parents = [copy.deepcopy(ind) for ind in self.population]
            parents.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'), reverse=True)
            elite = copy.deepcopy(parents[0])
            
            new_offspring = [elite]  # 保留最佳個體
            seen = {ind.genotype() for ind in parents}  # 用基因型簽名避免重複

            # Fallback 機制，避免無限迴圈
            max_trials = self.population_size * 20
            trials = 0

            # 進行繁殖直到滿足條件
            # Test
            cross_count = 0
            mutate_count = 0
            while len(new_offspring) < self.population_size and trials < max_trials:
                prev_len = len(new_offspring)
                # Step 2: 決定使用哪種操作
                operation = None
                if len(parents) == 1:
                    # Init: only used Point mutation 來增長族群
                    operation = 'point_mutation'
                else:
                    operation = random.choices(['crossover', 'point_mutation'], weights=[self.crossover_rate, self.point_mutation_rate])[0]

              
                # Step 3: 依據操作產生 offspring
                if operation == 'crossover':
                    # 進行交叉操作
                    parent1 = self.selection(parents)
                    parent2 = self.selection(parents)
                    childs = self.restrictedCrossover(parent1, parent2)  # 內部已 deepcopy，這裡不再 copy


                    if childs is None:
                        # Fallback: 若交叉失敗，則改用點突變
                        parent = self.selection(parents)
                        child = self.pointMutation(parent)
                        # 只加入不重複的個體 and check population size
                        g = child.genotype()
                        if g not in seen and len(new_offspring) < self.population_size:
                            seen.add(g)
                            new_offspring.append(child)

                    else:
                        # 只加入不重複的個體 and check population size
                        for child in childs:
                            if len(new_offspring) >= self.population_size:
                                break
                            g = child.genotype()
                            if g in seen:
                                continue
                            seen.add(g)
                            new_offspring.append(child)
                            cross_count += 1

               
                    
                elif operation == 'point_mutation':
                    # 進行點突變操作
                    parent = self.selection(parents)
                    child = self.pointMutation(parent)
                    g = child.genotype()
                    # 只加入不重複的個體 and check population size
                    if g not in seen and len(new_offspring) < self.population_size:
                        seen.add(g)
                        new_offspring.append(child)
                        mutate_count += 1

                else:
                    raise ValueError("Unknown operation")

                # Step 4: 避免重複個體 : 採用嚴格比對
                # 前述以比對過，這裡省略
                new_offspring_set = new_offspring
                # for i, child in enumerate(new_offspring):
                #     is_duplicate = False

                #     # 避免移除 elite
                #     if i == 0:
                #         new_offspring_set.append(child)
                #         continue

                #     for ind in parents:
                #         if self._identical_structure(child.tree, ind.tree, mode = 'strict'):
                #             is_duplicate = True
                #             print(f"[Warning] Duplicate individual detected and skipped. Child: {child.show()} matches Parent: {ind.show()}")
                #             break
                #     if not is_duplicate:
                #         new_offspring_set.append(child)
                   
                        

                if len(new_offspring_set) == prev_len:
                    
                    if trials!=0 and trials%5 ==0:
                        print(f"[Info] All new offspring are duplicates. Retrying... (Trial {trials}/{max_trials}) Current population size: {len(new_offspring_set)}")
                    trials += 1
                    continue  # 若全是重複，則重試
                else:
                    trials = 0  # 成功產生非重複個體，重置計數器
                new_offspring = new_offspring_set

            self.population = new_offspring

            # Step 5: 計算新族群的適應度
            # for ind in self.population:
            #     ind.evaluate(self.df, self.returns, self.fitness_type)
            self._evaluate_population(self.population)
            
            self.population.sort(key=lambda x: x.fitness if x.fitness else 0, reverse=True)
            best = self.population[0]
            best_ic = best.ic if best.ic is not None else 0.0
            best_sharpe = best.sharpe if best.sharpe is not None else 0.0
            best_fixed_r = best.fixed_r if best.fixed_r is not None else 0.0
            best_random_r = best.random_r if best.random_r is not None else 0.0

            self.best_history.append({
                'generation': gen + 1,
                'fitness': best.fitness,
                'ic': best_ic,
                'sharpe': best_sharpe,
                'fixed_r': best_fixed_r,
                'random_r': best_random_r
            })
            execution_time = time.time() - start_time
            execution_min = int(execution_time // 60)
            execution_sec = execution_time % 60

            print(f"Gen {gen+1}: {best.show()} Best Fitness={best.fitness:.4f}, IC={best_ic:.4f}, Sharpe={best_sharpe:.4f}, Fixed R={best_fixed_r:.4f}, Random R={best_random_r:.4f} Time={execution_min:.2f} min {execution_sec:.2f} sec , Cross={cross_count}, Mutate={mutate_count}, Population={len(self.population)}")

            # Early Stopping 檢查
            current_best_fitness = best.fitness if best.fitness else 0
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= tolerance_trials:
                print(f"No improvement in fitness for {tolerance_trials} consecutive generations. Stopping early at generation {gen+1}.")
                break

        # Step 6: 演化結束，回傳最佳個體
        self.population.sort(key=lambda x: x.fitness if x.fitness else 0, reverse=True)
        best = self.population[0]
        print(f"Final Best: {best.show_metrics()}")

        return best


alphas = [
    # Alpha 1: -1 * correlation(volume, close, N)---成交量与收盘价在N日内的背离程度
    OpNode('*',
        Leaf(-1),
        OpNode('correlation',
            Leaf('volume'),
            Leaf('close')
        )
    ),
    # Alpha 2: -1 * correlation(delta(volume, 1), delta(close,1), N)---成交量变动与收盘价日内变动在N日内的背离程度
    OpNode('*',
        Leaf(-1),
        OpNode('correlation',
            OpNode('delta', Leaf('volume'), Leaf(1)),
            OpNode('delta', Leaf('close'), Leaf(1))
        )
    ),
    # Alpha 3: -1 * correlation(rank(delta(volume), 1)), rank(delta(close,1)), N)
    OpNode('*',
        Leaf(-1),
        OpNode('correlation',
            OpNode('rank', OpNode('delta', Leaf('volume'), Leaf(1))),
            OpNode('rank', OpNode('delta', Leaf('close'), Leaf(1)))
        )
    ),
    # Alpha 4: correlation(close,open,N)
    OpNode('correlation',
        Leaf('close'),
        Leaf('open')
    ),
    # Alpha 5: (high + low)/2 - close
    OpNode('-',
        OpNode('/',
            OpNode('+', Leaf('high'), Leaf('low')),
            Leaf(2)
        ),
        Leaf('close')
    ),

    # Alpha 6: (2 * close - low - high) / ( high - low + 0.0001)
    OpNode('/',
        OpNode('-',
            OpNode('*', Leaf(2), Leaf('close')),
            OpNode('+', Leaf('low'), Leaf('high'))
        ),
        OpNode('+',
            OpNode('-',
                Leaf('high'),
                Leaf('low')
            ),
            Leaf(0.0001)
        )
    ),
]
    
# ================== 使用示例 ==================
def main(
        data_path: str = '../data/binanceusdm_swap_BTC-USDT-USDT_1h.csv',
        start_date: str = "2025-04-30 23:00:00+08:00",
        alphas: list[OpNode] | None = None,
        # generators
        population_size: int = 150,
        generations: int = 100,
        fitness_type: str = 'ic',  # DefaultEvaluator :'ic' 或 'sharpe' ; BiserialEvaluator: 'fixed' / 'random'
        evaluator: BaseEvaluator = DefaultEvaluator(), # 評估器: 可選 DefaultEvaluator 或其他自訂評估器
        depth: int = 5,
        point_mutation_rate: float = 0.4,
        crossover_rate: float = 0.7,
        tournament_size: int = 5,
        early_stopping_generations: int = 20,
        operator_set: list[str] | None = None,
        terminal_set: list[str] | None = None,
        ic_threshold: float = 0.02,
        sharpe_threshold: float = 0.9,
        # multiprocessing
        n_jobs: int = 30,  # 使用進程數量
        mp_start_method: str | None = 'spawn',    # ← 跨平台穩定；Linux 可不填或用 'fork'
        save_folder: str = "",
        ):
    
    # 生成測試數據
    # np.random.seed(42)
    # dates = pd.date_range('2020-01-01', periods=200, freq='D')
    # data = pd.DataFrame({
    #     'open': np.random.uniform(10, 15, 200),
    #     'close': np.random.uniform(10, 15, 200), 
    #     'high': np.random.uniform(12, 18, 200),
    #     'low': np.random.uniform(8, 12, 200),
    #     'volume': np.random.uniform(100, 500, 200)
    # }, index=dates)
    # import os 
    # print(f"可能核心數量 : {os.cpu_count()}")
    
    # """前處理"""
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    # data = pd.read_csv(data_path)
    data = data[data.index <= start_date]
    print(f"Data Range: {data.index.min()} to {data.index.max()}, Total Rows: {len(data)}")
    returns = data['close'].pct_change().shift(-1)
    

    #Add Features
    data[f"obv"] = talib.OBV(data['close'], data['volume'])
    for n in [5,10,20]:
        data[f"ema_{n}"] = talib.EMA(data['close'], timeperiod=n)
        data[f"rsi_{n}"] = talib.RSI(data['close'], timeperiod=n)
        data[f"atr_{n}"] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=n)
        data[f"cci_{n}"] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=n)
        data[f"mfi_{n}"] = talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=n)
        data[f"adx_{n}"] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=n)
        data[f"willr_{n}"] = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=n)
    feature_cols = data.columns.tolist()
    # data.to_csv(os.path.join(save_folder, "training_data_with_features.csv"),index=False)
    if 'timestamp' in feature_cols:
        feature_cols.remove('timestamp')  # 移除目標變數
    terminals = terminal_set if terminal_set is not None else feature_cols
    print("Feature columns:", feature_cols)

    """定義一些已知的 alpha 公式樹（可選）"""
    alphas_to_use = alphas if alphas is not None else globals().get('alphas', [])
    
    """Signature Test"""
    # test_ind = Individual((
    #     OpNode('*',
    #            OpNode(
    #                "rank",
    #                  OpNode(
    #                       'rolling_std',
    #                       Leaf('close'),
    #                       Leaf(10)
    #                  ),
    #                  Leaf(15)
    #            ),
    #             Leaf(-1)
    #     )
    # ))
    # print(f"Alpha Formula: {test_ind.show()}, Signature: {test_ind.signature}")
    # print(pformat(test_ind.node_to_canonical_dict(test_ind.tree), indent=2, width=50))
    # test_ind2 = Individual((
    #     OpNode('*', 
    #            OpNode(
    #                'rank', 
    #                Leaf(10),
    #                OpNode(
    #                    'rolling_std',
    #                     Leaf('close'),
    #                     Leaf(10)
    #                )
    #             ),
    #             Leaf(-1)
    #         )
    # ))

    # print(f"Alpha Formula: {test_ind2.show()}, Signature: {test_ind2.signature}")
    
    # print(pformat(test_ind2.node_to_canonical_dict(test_ind2.tree), indent=2, width=50))
    #pprint(f"dict: {test_ind2.node_to_canonical_dict(test_ind2.tree)}",indent=2, width=80, sort_dicts=False)
    # test_ind1 = OpNode('correlation',
    #                     Leaf('high'),
    #                     Leaf('volume'),
    #                 )
    # print(test_ind1.eval(data))
    # test_ind2 = OpNode('rank',
    #                         OpNode('rolling_std',
    #                             Leaf('high'),
    #                             Leaf(10)
    #                         )
    #                     )
    # print(test_ind2.eval(data))
    # test_ind3 = OpNode('*',test_ind2,Leaf(-1))
    # print(test_ind3.eval(data))
    # test_ind = OpNode('*',test_ind3,test_ind1)
    # print(test_ind.eval(data))
    # Test Alpha 101 : ***Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)) ***
    # alpha4_tree = OpNode('*',
    #                 OpNode('*',
    #                     Leaf(-1),
    #                     OpNode('rank',
    #                         OpNode('rolling_std',
    #                             Leaf('high'),
    #                             Leaf(10)
    #                         )
    #                     )
    #                 ),
    #                 OpNode('correlation',
    #                     Leaf('high'),
    #                     Leaf('volume'),
    #                 )
    #             )
    # test_individual = Individual(alpha4_tree)
    # fitness = test_individual.evaluate(data, returns, fitness_type='ic')
    # print("Alpha#4 Formula:", test_individual.show())
    # print(f"Alpha#4 Fitness: {fitness:.4f}, IC: {test_individual.ic:.4f}, Sharpe: {test_individual.sharpe:.4f}")

    
    # # # 執行遺傳演算法
    os.makedirs(save_folder, exist_ok=True)

    for i, ind in enumerate(alphas_to_use):
        test_individual = Individual(ind)
        print("Alpha Formula:", test_individual.show())
        print("Evaluator:", evaluator.__class__.__name__)

        solver_kwargs = dict(
            df=data,
            returns=returns,
            population_size=population_size,
            generations=generations,
            fitness_type=fitness_type,  # 可選 'ic' 或 'sharpe'
            evaluator=evaluator,
            population=[test_individual],  # 初始種群:只能放一個
            depth=depth,
            point_mutation_rate=point_mutation_rate,
            crossover_rate=crossover_rate,
            tournament_size=tournament_size,
            terminal_set=terminals,  # 使用擴展後的特徵集
            early_stopping_generations=early_stopping_generations,
            # NEW multiprocessing
            n_jobs=n_jobs,
            mp_start_method=mp_start_method,    # ← 跨平台穩定；Linux 可不填或用 'fork'
        )
        if operator_set is not None:
            solver_kwargs["operator_set"] = operator_set

        solver = GeneticAlphaSolver(**solver_kwargs)

        best_alpha = solver.evolve()
        print(f"\n最佳Alpha {best_alpha.show()} - {best_alpha.show_metrics()}\n")

        if abs(best_alpha.fitness) >= 0.15:
            # 儲存最佳Alpha
            save_path = os.path.join(save_folder, f"best_evolved_alpha_from_predefined_{i}.json")
            save_alpha(best_alpha, save_path)

    # 儲存最佳Alpha    
    alpha_path = os.path.join(save_folder, "best_evolved_alpha.json")
    save_alpha(best_alpha, alpha_path)
    # 載入最佳Alpha
    loaded_alpha = load_alpha(alpha_path)
    print(f"載入的Alpha: {loaded_alpha.show()} - {loaded_alpha.show_metrics()}")

    """HOW TO USE"""
    # return_features  = loaded_alpha.tree.eval(your_dataframe)

if __name__ == "__main__":
    df = pd.read_csv("/home/louisexpc/Crypto-Quant-Monitor/train/data/binanceusdm_swap_BTC-USDT-USDT_1h.csv") 
    event_df = pd.read_csv("/home/louisexpc/Crypto-Quant-Monitor/train/data/BTC-USDT_1h_ewma_up8_dn10_lookback108_label.csv")      
    evaluator = BiserialRankEvaluator(event_df=event_df, df=df, lookback=108)

    main(
        data_path="/home/louisexpc/Crypto-Quant-Monitor/train/data/binanceusdm_swap_BTC-USDT-USDT_1h.csv",
        evaluator=evaluator,
        start_date="2025-04-30 23:00:00+08:00",
        population_size=150,
        generations=2,
        depth=10,
        fitness_type='fixed',
        n_jobs=8,
        save_folder="./results")
    
    """TODO: follow, debug 中記得繼續回去補充"""
    # test1 = Individual(
    #     OpNode('*',
    #         Leaf(-1),
    #         OpNode('correlation',
    #             Leaf('low'),
    #             Leaf('close')
    #         )
    #     )
    # )
    # test2 = Individual(
    #     OpNode('+',
    #         Leaf(-1),
    #         OpNode('correlation',
    #             Leaf('volume'),
    #             Leaf('high')
    #         )
    #     )
    # )
    # from pprint import pprint, pformat
    # print(pformat(test1.node_to_canonical_dict(test1.tree), indent=2, width=50))
    # print(pformat(test2.node_to_canonical_dict(test2.tree), indent=2, width=50))
    # solver = GeneticAlphaSolver(
    #         df = pd.DataFrame(),
    #         returns = pd.Series(),
    #         n_jobs=1
    #     )
    # print(solver._identical_structure_update(test1, test2))
    