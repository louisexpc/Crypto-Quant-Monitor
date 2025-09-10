# 基於遺傳演算法的隨機Alpha生成器
# Ref : https://arxiv.org/html/2412.00896v1


import pandas as pd
import numpy as np
import random
import copy
from typing import Any, List, Tuple
from node import Node, Leaf, OpNode


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

def calc_ic(signal, returns):
    """計算信息系數IC"""
    if signal is None or signal.isnull().all():
        return np.nan
    combined = pd.concat([signal, returns], axis=1).dropna()
    if combined.empty or len(combined) < 10:
        return np.nan
    return combined.iloc[:,0].corr(combined.iloc[:,1])

def calc_sharpe(signal, returns):
    """計算基於信號的策略夏普比率"""
    if signal is None or signal.isnull().all():
        return np.nan
    
    # 簡化策略：信號>0做多，<0做空
    positions = np.sign(signal.fillna(0))
    strategy_returns = positions.shift(1) * returns
    strategy_returns = strategy_returns.dropna()
    
    if len(strategy_returns) == 0:
        return np.nan
    
    return strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

# ================== 遺傳演算法組件 ==================

class Individual:
    """個體：一個alpha公式樹"""
    def __init__(self, tree:Node):
        self.tree = tree
        self.fitness = None
        self.ic = None
        self.sharpe = None

    def evaluate(self, df, returns, fitness_type='ic'):
        """評估個體適應度"""
        try:
            signal = self.tree.eval(df)
            self.ic = calc_ic(signal, returns)
            self.sharpe = calc_sharpe(signal, returns)
            
            if fitness_type == 'ic':
                self.fitness = abs(self.ic) if not np.isnan(self.ic) else 0
            elif fitness_type == 'sharpe':
                self.fitness = self.sharpe if not np.isnan(self.sharpe) else 0
            else:
                # 組合評分
                ic_score = abs(self.ic) if not np.isnan(self.ic) else 0
                sharpe_score = self.sharpe if not np.isnan(self.sharpe) else 0
                self.fitness = ic_score * 0.5 + sharpe_score * 0.5
                
        except Exception as e:
            self.fitness = 0
        
        return self.fitness

class GeneticAlphaSolver:
    """遺傳演算法主體"""
    def __init__(
            self, 
            df:pd.DataFrame, 
            returns:pd.Series, 
            generations:int=10, 
            fitness_type:str='ic', 
            point_mutation_rate:float=0.3, 
            population_size:int =20, 
            depth:int=3,
            population = None | list[Individual],
            operator_set : list[str] = ["+", "-", "*", "/", "sqrt", "log", "inverse", "sigmoid",
                    "rank", "scale", "signedpower", "delay", "covariance", "correlation", "delta",
                    "decay_linear", "ts_stddev", "ts_sum", "ts_argmax", "ts_argmin", "ts_product",
                    "ts_rank", "ts_max", "ts_min", "ts_mean", "ts_wma", "ts_high", "ts_low"],
            terminal_set : list[str] = ["open", "close", "high", "low", "volume"]
        ):
        self.df = df
        self.returns = returns
        self.population_size = population_size
        self.generations = generations
        self.fitness_type = fitness_type
        self.point_mutation_rate = point_mutation_rate
        self.best_history = []
        self.operator_set = operator_set
        self.terminal_set = terminal_set
        self.depth = depth

        """
        根據論文描述，初始種群應該是簡單結構的 alpha 公式，若沒有指定，則隨機生成
        """
        if population is None:
            self.population = [Individual(random_tree()) for _ in range(population_size)]
        else:
            self.population = population



    def restrictedCrossover(self, parent1:Individual, parent2:Individual)-> Tuple[Individual, Individual]:
        # 1) 檢查整體結構相容（呼叫你已有的 _identical_structure）
        if not self._identical_structure(parent1.tree, parent2.tree):
            return parent1, parent2

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
            return child1, child2

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
        if new_tree1.tree_depth() > self.depth or new_tree2.tree_depth() > self.depth:
            # 拒絕此次 crossover（或做 retry 機制）
            return parent1, parent2  # 或 return child1, child2 unchanged

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
    
    def _identical_structure(self, node1:Node, node2:Node):
        """檢查兩個子樹是否結構相同: 
        簡化版，僅適用當前的運算符集(不超過二元的運算符)，未來拓展需要考量更多情況：
        - 葉節點（Leaf）要比較 terminal_role，而非單純回傳 True
        - child_roles 不應該只比對等於，而是做「相容性」判斷
        - 不要硬綁 binary（left/right）
        - 加入 strict/relaxed 模式
        - 交叉前後應有後驗驗證（validator）
        - 避免只比對 child_roles 的 list 等於
        """
        if type(node1) != type(node2):
            return False
        if isinstance(node1, Leaf):
            if  type(node1.value) == type(node2.value):
                return True
            else:
                return False
            
        if isinstance(node1, OpNode):

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
        self._pointMutation(individual.tree)

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
                



    def selection(self, k=3):
        """錦標賽選擇"""
        candidates = random.sample(self.population, k)
        return max(candidates, key=lambda x: x.fitness if x.fitness else 0)

    def evolve(self):
        """主進化循環
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
        for gen in range(self.generations):
            # 評估所有個體
            # 在論文中預設初始只會有一個
            # 實務上這邊應該要平行化
            if len(self.population) == 1 :
                ind = self.population[0]
                if gen == 0 or gen== 1:
                    # 初始兩代只能突變
                    ind.evaluate(self.df, self.returns, self.fitness_type)
                    print(f"Gen {gen+1}: Fitness={ind.fitness:.4f}, IC={ind.ic:.4f}, Sharpe={ind.sharpe:.4f}")
                    new_population = [ind]
                    while len(new_population) < self.population_size:
                        parent = self.selection()
                        child = copy.deepcopy(parent)
                        self.pointMutation(child)
                        new_population.append(child)
                    self.population = new_population
            else:
                # 正常的種群迭代
                for ind in self.population:
                    ind.evaluate(self.df, self.returns, self.fitness_type)
                
                # 排序並記錄最佳個體
                self.population.sort(key=lambda x: x.fitness if x.fitness else 0, reverse=True)
                best = self.population[0]
                self.best_history.append({
                    'generation': gen + 1,
                    'fitness': best.fitness,
                    'ic': best.ic,
                    'sharpe': best.sharpe
                })
                
                print(f"Gen {gen+1}: Best Fitness={best.fitness:.4f}, IC={best.ic:.4f}, Sharpe={best.sharpe:.4f}")
                
                # 生成新種群
                new_population = self.population[:self.population_size//4]  # 保留菁英
                
                while len(new_population) < self.population_size:
                    parent1 = self.selection()
                    parent2 = self.selection()
                    child = self.crossover(parent1, parent2)
                    
                    if random.random() < self.mutation_rate:
                        self.mutate(child)
                    
                    new_population.append(child)
                
                self.population = new_population
        
        return self.population[0]

# ================== 使用示例 ==================

if __name__ == "__main__":
    # 生成測試數據
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    data = pd.DataFrame({
        'open': np.random.uniform(10, 15, 200),
        'close': np.random.uniform(10, 15, 200), 
        'high': np.random.uniform(12, 18, 200),
        'low': np.random.uniform(8, 12, 200),
        'volume': np.random.uniform(100, 500, 200)
    }, index=dates)
    
    returns = data['close'].pct_change().shift(-1)
    
    # 執行遺傳演算法
    solver = GeneticAlphaSolver(
        df=data, 
        returns=returns, 
        population_size=15, 
        generations=5,
        fitness_type='ic'
    )
    
    best_alpha = solver.evolve()
    print(f"\n最佳Alpha - IC: {best_alpha.ic:.4f}, Sharpe: {best_alpha.sharpe:.4f}")
