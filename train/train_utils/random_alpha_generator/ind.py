# ================== 遺傳演算法組件 ==================
from node import Node, Leaf, OpNode
from evaluation import calc_ic, calc_sharpe, DefaultEvaluator
import json, time
import hashlib
import numpy as np
from typing import Any

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


class Individual:
    """個體：一個alpha公式樹"""
    def __init__(self, tree:Node):
        self.tree = tree
        self.fitness = None
        self.ic = None
        self.sharpe = None
        self.evaluator = None
        # print(f"[Debug] Initializing New Individual Module.")


        """Operater Info: 目前支援項目"""
        self.COMMUTATIVE_OPS = {"+", "*", "correlation", "covariance"}  # 若未來要擴充，照樣加入
        self.WINDOW_OPS = {"rolling_mean", "rolling_std", "signedpower", "delay",
                    "delta", "decay_linear", "ts_stddev", "ts_sum", "ts_argmax",
                    "ts_argmin", "ts_product", "ts_rank", "ts_max", "ts_min",
                    "ts_mean", "ts_wma", "ts_highday", "ts_lowday"}
        
        self.original_signature = self.genotype()          # 原始結構簽名: 初始化後不變
        self.signature = self.original_signature           # 結構簽名:隨演化突變
    def show_metrics(self)->str:
        """
        Display the non-None evaluation metrics of the individual.
        Metrics displayed:
            - Fitness: The fitness value of the individual, if available.
            - IC: The information coefficient, if available.
            - Sharpe: The Sharpe ratio, if available.
        Returns:
            str: A comma-separated string of key:value pairs for each non-None metric.
        """
        metrics = []
        if self.fitness is not None:
            metrics.append(f"Fitness: {self.fitness}")
        if self.ic is not None:
            metrics.append(f"IC: {self.ic}")
        if self.sharpe is not None:
            metrics.append(f"Sharpe: {self.sharpe}")
        return ", ".join(metrics)

    def evaluate(self, df, returns, fitness_type='ic'):
        """評估個體適應度"""
        ev = self.evaluator
        if ev is None:
            ev = DefaultEvaluator()
        # else:
        #     print(f"[Debug] Using custom evaluator: {type(ev)}")    
        
        self.fitness, metrics = ev.evaluate(self, df, returns, fitness_type)
        self.ic = metrics.get("ic", None)
        self.sharpe = metrics.get("sharpe", None)
        self.fixed_r = metrics.get("fixed_r", None)
        self.random_r = metrics.get("random_r", None)

        return self.fitness
        

    
    def show(self) -> str:
        """
        將 alpha 公式樹轉為可閱讀字串（中序表示）
        """
        def _fmt_const(v):
            # 盡量印成整數樣式；否則用短格式浮點
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            if isinstance(v, (float, np.floating)):
                return str(int(v)) if float(v).is_integer() else f"{float(v):.6g}"
            return str(v)

        def _visit(node: Node) -> str:
            from typing import Optional
            if isinstance(node, Leaf):
                return node.value if isinstance(node.value, str) else _fmt_const(node.value)

            if isinstance(node, OpNode):
                op = node.operator
                arity = node.arity

                # --- 算術（中序，強制括號以保結構） ---
                if op in {"+", "-", "*", "/"}:
                    left = _visit(node.left)
                    right = _visit(node.right)
                    return f"({left} {op} {right})"

                # --- 一元函數：函數式表示 ---
                if arity == 1:
                    return f"{op}({_visit(node.left)})"

                # --- 需要 window 的二元（series, window） ---
                win_ops = {
                    "rolling_mean","rolling_std",
                    "signedpower","delay",
                    "delta","decay_linear","ts_stddev","ts_sum","ts_argmax","ts_argmin",
                    "ts_product","ts_rank","ts_max","ts_min","ts_mean","ts_wma",
                    "ts_highday","ts_lowday",
                }
                if op in win_ops:
                    return f"{op}({_visit(node.left)}, {_visit(node.right)})"

                # --- 雙變量統計（series, series[, maybe window]) ---
                if op in {"covariance", "correlation"}:
                    left = _visit(node.left)
                    right = _visit(node.right)
                    # 若樹節點真的有第三參數（可選）就一併輸出
                    third = getattr(node, "third", None)
                    if isinstance(third, Node) and third is not None:
                        return f"{op}({left}, {right}, {_visit(third)})"
                    return f"{op}({left}, {right})"

                # --- 其它：通用函數式回退 ---
                args = []
                if node.left is not None:
                    args.append(_visit(node.left))
                if node.right is not None:
                    args.append(_visit(node.right))
                return f"{op}(" + ", ".join(args) + ")"

            # 不期望型別
            return "<?>"

        return _visit(self.tree)
    
    def to_dict(self):
        return {
            "tree": node_to_dict(self.tree),
            "metrics": {
                "fitness": self.fitness,
                "ic": self.ic,
                "sharpe": self.sharpe,
            },
            "meta": {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fitness_type": getattr(self, "fitness_type", None),
                "operator_set": getattr(self, "operator_set", None),
                "depth_limit": getattr(self, "depth", None),
                "signature": getattr(self, "signature", None),  # 有的話
            },
            "formula_str": self.show(),  # 方便人看
        }

    @classmethod
    def from_dict(cls, d):
        ind = cls(node_from_dict(d["tree"]))
        # 指標在重新 evaluate 後會更新；也可先灌進來供顯示
        m = d.get("metrics", {})
        ind.fitness = m.get("fitness", None)
        ind.ic = m.get("ic", None)
        ind.sharpe = m.get("sharpe", None)
        ind.signature = d.get("meta", {}).get("signature", None)
        return ind
    # Signature 相關
    def genotype(self)->str:
        """API: 取得基因型字串表示"""
        return self.genotype_signature(self.tree)
    def genotype_signature(self, node:Node) -> str:
        """
        取得基因型簽名: 只考慮運算符與結構，不考慮常數與欄位
        Args:
        - node: 樹節點
        Returns:
        - str: 基因型簽名字串
        """
        return self._digest_dict(self.node_to_canonical_dict(node))

    def _normalize_const(self, val: float, ndigits: int = 8) -> float:
        """
        將常數標準化為固定小數位數，避免浮點誤差影響結構簽名
        Args:
        - val: 常數值
        - ndigits: 小數位數
        Returns:
        - 標準化後的常數
        """
        try:
            return round(float(val), ndigits)
        except Exception:
            return float(val)
    def _is_scalar_leaf(self, node:Node) -> bool:
        """
        判斷是否為純數值常數葉節點: Leaf 節點且 value 非字串
        Args:
        - node: 樹節點
        Returns:
        - 是否為純數值常數葉節點
        """
        return isinstance(node, Leaf) and not isinstance(node.value, str)
    
    def _is_field_leaf(self, node:Node) -> bool:
        """
        判斷是否為欄位葉節點: Leaf 節點且 value 為字串
        Args:
        - node: 樹節點
        Returns:
        - 是否為欄位葉節點
        """
        return isinstance(node, Leaf) and isinstance(node.value, str)
    
    def _canonical_leaf(self,node:Node) -> dict[str, Any]:
        """
        將 Leaf 節點轉為標準化 dict 表示
        Args:
        - node: 樹節點
        Returns:
        - dict 表示
        """
        assert isinstance(node, Leaf)
        if isinstance(node.value, str):
            # 欄位名 Leaf：直接用字串
            return {"type": "Leaf", "kind": "field", "value": node.value}
        else:
            # 常數 Leaf：規格化
            return {"type": "Leaf", "kind": "const", "value": self._normalize_const(node.value)}
        
    def node_to_canonical_dict(self,node:Node) -> dict[str, Any]:
        """
        把樹轉為「穩定的、可交換一致、window 語義固定」的 dict 表示。
        - 對 commutative ops：依子樹簽名排序左右子，消除 (a+b) vs (b+a) 差異。
        - 對 window 類：強制 left 為 series, right 為 window（若輸入顛倒，這裡重排）。
        """
        if isinstance(node, Leaf):
            return self._canonical_leaf(node)

        assert isinstance(node, OpNode)
        op = node.operator  # operator: string
        ar = node.arity     # arity: 1 or 2

        # 先遞迴拿到左右的 canonical dict（暫時不排）
        left_d  = self.node_to_canonical_dict(node.left) if node.left  is not None else None
        right_d = self.node_to_canonical_dict(node.right) if node.right is not None else None

        # --- commutative ops: 排序子樹 ---
        if op in self.COMMUTATIVE_OPS and left_d is not None and right_d is not None:
            # 以子樹簽名排序：確保 (a+b) 與 (b+a) 的序列化一致
            lh = self._digest_dict(left_d) # 左子樹雜湊: str
            rh = self._digest_dict(right_d) # 右子樹雜湊: str
            if rh < lh:  # 比字典序小就互換
                left_d, right_d = right_d, left_d
        
        # ----- window 類正規化：左=series, 右=window -----
        if op in self.WINDOW_OPS and left_d is not None and right_d is not None:
            # 判斷 series vs window：
            #   series：欄位 leaf 或經運算的子樹（OpNode）
            #   window：常數 leaf（注意：你也可能用「常數 series」但在樹上會是 Leaf 常數）
            def _is_series_like(d):
                return (d["type"] == "Leaf" and d.get("kind") == "field") or (d["type"] == "OpNode")
            def _is_window_like(d):
                return (d["type"] == "Leaf" and d.get("kind") == "const")
            
            if _is_window_like(left_d) and _is_series_like(right_d):
                # 交換，固定 series 在左
                left_d, right_d = right_d, left_d
        
        # --- 建立本節點 dict ---
        out = {
            "type": "OpNode",
            "op": op,
            "op_class": node.node_class,
            "arity": ar,
            "left": left_d,
            "right": right_d
        }

        return out
    # Hash Helper Function
    def _digest_dict(self,d: dict[str, Any]) -> str:
        """
        將 dict 轉為 JSON（鍵排序）後做雜湊，回傳短字串（簽名的「原料」）。
        也可直接回傳 JSON 字串做 key（0 碰撞），但用雜湊可省記憶體。
        Args:
        - d: dict, 可巢狀
        Returns:
        - str: 簽名字串
        """
        s = json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        # blake2b 速度快，可調 digest_size 短一點
        return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()