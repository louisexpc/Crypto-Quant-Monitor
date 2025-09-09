# reddit_sent_to_vector.py  (with sarcasm + tqdm + robust fallback)
import os, json, re, math
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

"""
1. datetime: UTC 的「窗格收盤時間」（不是起點）。例如 15:30 代表 [15:15, 15:30) 這個 15 分鐘窗的結束點。

2. timestamp: 對應 datetime 的 Unix 毫秒（int）。方便下游和行情時間對齊。

3. sent_pos_neg_spread: 公式：mean(pos) - mean(neg)。
    +1 越多頭、-1 越空頭。0 附近中性。
    這是你交易上最直觀的情緒多空訊號。

4. sent_entropy: 公式：-∑ p_k * log(p_k)，k ∈ {pos, neu, neg}。
    範圍：[0, ln 3≈1.099]。越低＝越「一致」（模型更確信某一類），越高＝越分歧／不確定。

5. sent_n_texts: 這一窗內計入的文本數（submission+comments）。
    越小越不穩（建議做最小樣本數門檻或下游加權）。

6/7 . sent_pos_mean / sent_neg_mean: 該窗所有文本「正向／負向機率」的 平均值（來自 FinBERT softmax 機率）。

8. sent_sarcasm_ratio（新增）
    反串比例＝該窗被諷刺/反串模型判為正類的文本數 / N。

    越高表示語氣常「反諷」。高比率＋小 N → 高風險。

9. sent_pos_mean_sans_sarcasm / sent_neg_mean_sans_sarcasm（新增）
    把被判為反串的文本 剔除 後，重新計平均。
    若整窗都是反串（ratio=1），為避免 NaN，我們保守地回填成原平均（=不變）。

10. sent_pos_neg_spread_sans_sarcasm: pos_sans - neg_sans，也就是「剔除反串」後的多空差。

可拿來和原版 spread 做 A/B，評估反串處理的淨效果。
"""


cfg = {
    "jsonl_files": [
        "data/reddit/bitcoin.jsonl",
        "data/reddit/ethereum.jsonl",
        # "data/reddit/solana.jsonl",
        # "data/reddit/dogecoin.jsonl",
        "data/reddit/BNBinance.jsonl",
        # "data/reddit/pepecoin.jsonl",
    ],
    "window": "15T",
    "start_utc": "2024-01-01T00:00:00+00:00",
    "end_utc":   "2099-12-31T23:59:59+00:00",
    "min_words": 3,
    "batch_size": 1024,
    "sent_model": "ProsusAI/finbert",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_dataparallel": False,
    "out_csv": "data/reddit/reddit_sentiments_with_sarcasm.csv",

    # === Sarcasm/Irony ===
    # 首選 + 備援（依序嘗試；載不到就跳下一個）
    "use_sarcasm": True,
    "sarcasm_model_candidates": [
        "cardiffnlp/twitter-roberta-base-irony",   # 推薦：TweetEval 的諷刺偵測
        "helinivan/english-sarcasm-detector",      # 英文新聞標題
        "helinivan/multilingual-sarcasm-detector", # 多語
    ],
    "sarcasm_threshold": 0.70,  # ≥ 閾值 → 視為反串/諷刺
}

URL_PAT = re.compile(r"https?://|www\\.")
WS_PAT  = re.compile(r"\\s+")
MD_PAT  = re.compile(r"[`*_>#|\\[\\]\\(\\)]")

def safe_text(x) -> str:
    if x is None: return ""
    if isinstance(x, float):
        if math.isnan(x): return ""
        return str(x)
    if isinstance(x, (int, bool)): return str(x)
    if isinstance(x, str): return x
    if isinstance(x, list): return " ".join(safe_text(t) for t in x)
    return ""

def clean_text(t: str) -> str:
    t = (t or "").strip()
    t = MD_PAT.sub(" ", t)
    t = WS_PAT.sub(" ", t)
    return t

def pick_submission_text(rec: Dict) -> str:
    title = safe_text(rec["title"]) if "title" in rec else ""
    selftext = safe_text(rec["selftext"]) if "selftext" in rec else ""
    return (title + " " + selftext).strip()

def to_int_ts(x) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None

def iter_records(paths: List[str]):
    # 檔案層級 tqdm
    for path in tqdm(paths, desc="Reading JSONL files", unit="file"):
        with open(path, "r", encoding="utf-8") as f:
            # 行層級 tqdm（不預先計數，避免二次讀檔）
            for line in tqdm(f, desc=f"Parsing {Path(path).name}", unit="lines", leave=False):
                if not line.strip():
                    continue
                rec = json.loads(line, parse_constant=lambda _x: None)

                # submission
                ts = to_int_ts(rec["created_utc"] if "created_utc" in rec else 0)
                if ts is not None:
                    text = clean_text(pick_submission_text(rec))
                    if text and len(text.split()) >= int(cfg["min_words"]):
                        yield ts, text

                # comments
                comments = rec["comments"] if "comments" in rec else []
                if isinstance(comments, list):
                    for c in comments:
                        c = c or {}
                        cts = to_int_ts(c["created_utc"] if "created_utc" in c else 0)
                        if cts is None:
                            continue
                        body = clean_text(safe_text(c["body"] if "body" in c else ""))
                        if body and len(body.split()) >= int(cfg["min_words"]):
                            yield cts, body

# --- FinBERT ---
tok = AutoTokenizer.from_pretrained(cfg["sent_model"])
clf = AutoModelForSequenceClassification.from_pretrained(cfg["sent_model"]).to(cfg["device"])
if cfg["use_dataparallel"] and torch.cuda.device_count() > 1:
    clf = torch.nn.DataParallel(clf)

@torch.no_grad()
def texts_to_probs(texts: List[str]) -> np.ndarray:
    # FinBERT: 0=neg,1=neu,2=pos → 轉為 [pos,neu,neg]
    b = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=192).to(cfg["device"])
    logits = clf(**b).logits
    probs = torch.softmax(logits, dim=-1)[:, [2,1,0]]
    return probs.detach().cpu().numpy()

# --- Sarcasm/Irony Loader（自動 fallback） ---
sar_tok = None
sar_clf = None
_sar_idx = None  # 正類（sarc/irony）索引

def _try_load_sarcasm(model_id: str) -> bool:
    global sar_tok, sar_clf, _sar_idx
    try:
        sar_tok = AutoTokenizer.from_pretrained(model_id)
        sar_clf = AutoModelForSequenceClassification.from_pretrained(model_id).to(cfg["device"])
        if cfg["use_dataparallel"] and torch.cuda.device_count() > 1:
            sar_clf = torch.nn.DataParallel(sar_clf)

        id2label = sar_clf.module.config.id2label if isinstance(sar_clf, torch.nn.DataParallel) else sar_clf.config.id2label
        # 從 label 名稱自動找出「sarcasm/irony」的 index
        _sar_idx = None
        for k, v in id2label.items():
            name = (v if isinstance(v, str) else str(v)).lower()
            if ("sarc" in name) or ("irony" in name) or ("ironic" in name):
                _sar_idx = int(k)
                break
        if _sar_idx is None:
            # 若為二分類且未標出名稱，慣例用 1 當正類
            _sar_idx = 1 if len(id2label) == 2 else 0
        tqdm.write(f"[sarcasm] loaded: {model_id} (positive_index={_sar_idx}, labels={id2label})")
        return True
    except Exception as e:
        tqdm.write(f"[sarcasm] failed to load {model_id}: {e}")
        sar_tok, sar_clf, _sar_idx = None, None, None
        return False

if bool(cfg["use_sarcasm"]):
    loaded = False
    for mid in cfg["sarcasm_model_candidates"]:
        if _try_load_sarcasm(mid):
            loaded = True
            break
    if not loaded:
        tqdm.write("[sarcasm] all candidates failed; continuing WITHOUT sarcasm.")

@torch.no_grad()
def texts_to_sarcasm_prob(texts: List[str]) -> np.ndarray:
    # 若未載入，回傳全 0
    if sar_tok is None or sar_clf is None or _sar_idx is None:
        return np.zeros((len(texts),), dtype=np.float32)
    b = sar_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=192).to(cfg["device"])
    logits = sar_clf(**b).logits
    probs = torch.softmax(logits, dim=-1)
    sar_p = probs[:, _sar_idx]
    return sar_p.detach().cpu().numpy()

def aggregate_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    df = df.copy()
    df["dt_start"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.floor(window)
    win_delta = pd.to_timedelta(window)
    df["dt_end"] = df["dt_start"] + win_delta

    # 全部文本
    agg_all = df.groupby("dt_end").agg(
        sent_pos_mean=("p_pos", "mean"),
        sent_neg_mean=("p_neg", "mean"),
        sent_neu_mean=("p_neu", "mean"),
        sent_n_texts=("p_pos", "size"),
    ).reset_index()

    # spread / 熵
    eps = 1e-12
    p_all = agg_all[["sent_pos_mean","sent_neu_mean","sent_neg_mean"]].to_numpy(float)
    spread_all = p_all[:, 0] - p_all[:, 2]
    ent_all = -np.sum(p_all * np.log(p_all + eps), axis=1)

    out = agg_all[["dt_end"]].copy()
    out["sent_pos_neg_spread"] = spread_all.astype(np.float32)
    out["sent_entropy"] = ent_all.astype(np.float32)
    out["sent_n_texts"] = agg_all["sent_n_texts"].astype(np.int32)
    out["sent_pos_mean"] = agg_all["sent_pos_mean"].astype(np.float32)
    out["sent_neg_mean"] = agg_all["sent_neg_mean"].astype(np.float32)

    # 若有 sarcasm
    if "is_sarcasm" in df.columns:
        sar_ratio = df.groupby("dt_end")["is_sarcasm"].mean().reset_index().rename(columns={"is_sarcasm": "sent_sarcasm_ratio"})
        df_ns = df.loc[df["is_sarcasm"] == 0]
        agg_ns = df_ns.groupby("dt_end").agg(
            sent_pos_mean_sans_sarcasm=("p_pos", "mean"),
            sent_neg_mean_sans_sarcasm=("p_neg", "mean"),
        ).reset_index()

        out = out.merge(sar_ratio, on="dt_end", how="left")
        out = out.merge(agg_ns, on="dt_end", how="left")

        out["sent_pos_mean_sans_sarcasm"] = out["sent_pos_mean_sans_sarcasm"].fillna(out["sent_pos_mean"])
        out["sent_neg_mean_sans_sarcasm"] = out["sent_neg_mean_sans_sarcasm"].fillna(out["sent_neg_mean"])
        out["sent_pos_neg_spread_sans_sarcasm"] = (
            out["sent_pos_mean_sans_sarcasm"] - out["sent_neg_mean_sans_sarcasm"]
        ).astype(np.float32)
        out["sent_sarcasm_ratio"] = out["sent_sarcasm_ratio"].fillna(0.0).astype(np.float32)

    # 加上時間欄
    out["datetime"]  = out["dt_end"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    out["timestamp"] = (out["dt_end"].astype("int64") // 10**6).astype(np.int64)
    out = out.drop(columns=["dt_end"])

    base_cols = ["datetime","timestamp","sent_pos_neg_spread","sent_entropy",
                 "sent_n_texts","sent_pos_mean","sent_neg_mean"]
    extra_cols = []
    if "sent_sarcasm_ratio" in out.columns:
        extra_cols = ["sent_sarcasm_ratio",
                      "sent_pos_mean_sans_sarcasm",
                      "sent_neg_mean_sans_sarcasm",
                      "sent_pos_neg_spread_sans_sarcasm"]
    return out[base_cols + extra_cols].sort_values("timestamp").reset_index(drop=True)

def main():
    # 1) 讀取 & 篩時間
    rows_ts, rows_txt = [], []
    for ts, text in iter_records(cfg["jsonl_files"]):
        rows_ts.append(ts); rows_txt.append(text)
    if not rows_ts:
        print("No records found. Check your jsonl paths.")
        return

    df = pd.DataFrame({"ts": rows_ts, "text": rows_txt})
    s_utc = pd.to_datetime(cfg["start_utc"], utc=True)
    e_utc = pd.to_datetime(cfg["end_utc"], utc=True)
    mask = (pd.to_datetime(df["ts"], unit="s", utc=True) >= s_utc) & \
           (pd.to_datetime(df["ts"], unit="s", utc=True) <= e_utc)
    df = df.loc[mask].reset_index(drop=True)
    if len(df) == 0:
        print("No records in the specified time range.")
        return

    # 2) FinBERT 批次推論
    probs_all, B = [], int(cfg["batch_size"])
    for i in tqdm(range(0, len(df), B),
                  total=(len(df)+B-1)//B, desc="FinBERT inference", unit="batch"):
        probs_all.append(texts_to_probs(df["text"].iloc[i:i+B].tolist()))
    probs = np.vstack(probs_all)
    df["p_pos"] = probs[:,0]; df["p_neu"] = probs[:,1]; df["p_neg"] = probs[:,2]

    # 2.5) Sarcasm 批次推論（若有載入成功才執行）
    if bool(cfg["use_sarcasm"]) and (sar_tok is not None) and (sar_clf is not None):
        sar_all = []
        for i in tqdm(range(0, len(df), B),
                      total=(len(df)+B-1)//B, desc="Sarcasm inference", unit="batch"):
            sar_all.append(texts_to_sarcasm_prob(df["text"].iloc[i:i+B].tolist()))
        sar = np.concatenate(sar_all, axis=0)
        df["sar_prob"] = sar.astype(np.float32)
        df["is_sarcasm"] = (df["sar_prob"] >= float(cfg["sarcasm_threshold"])).astype(np.int8)

    # 3) 聚合 & 輸出
    out = aggregate_window(df, cfg["window"])
    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg["out_csv"], index=False)
    print(f"[OK] saved: {cfg['out_csv']}  shape={out.shape}")

if __name__ == "__main__":
    main()
