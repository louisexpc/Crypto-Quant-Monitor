# biance_trades.py
"""
下載 2023-01-01 起 BTCUSDT 的 逐日 trades 並合併成一個 CSV
"""
import os, io, zipfile, requests
import pandas as pd
from datetime import date, timedelta

BASE = "https://data.binance.vision"
PREFIX = r"data/futures/um/daily/trades/BTCUSDT"
OUTDIR = r"data/binance_trades"

def dl_one(day: date, outdir = OUTDIR):
    """
    Binance: 
    用來下載某一天的 BTC/USDT 交易資料的壓縮檔案（.zip）
    """
    outdir = os.path.join(outdir, "BTCUSDT")
    os.makedirs(outdir, exist_ok=True)
    fname = f"BTCUSDT-trades-{day.isoformat()}.zip" # day.isoformat() → 轉成 YYYY-MM-DD 字串
    url = f"{BASE}/{PREFIX}/{fname}"                # 組合成下載網址
    dst = os.path.join(outdir, fname)
    if os.path.exists(dst):
        return dst, False
    
    r = requests.get(url, timeout=60)               # 向組合好的 URL 發送 HTTP GET 請求
    if r.status_code == 200:                        # 200: 成功取得資料
        with open(dst, "wb") as f: f.write(r.content)
        return dst, True
    return None, False


def unzip_csvs(zpath):
    """
    每次解壓縮一個.zip => csv: 

    tradeId	該筆交易的 ID
    price	成交價格
    qty	成交數量（幣數）
    quoteQty	該筆交易的 quote value (USDT)
    time	時間戳記（通常是毫秒）
    isBuyerMaker	是不是 maker（掛單者）是買方？
    isBestMatch	是否為最優撮合結果？
    """
    with zipfile.ZipFile(zpath, "r") as z:
        for n in z.namelist():          # 列出 zip 裡所有的檔名（檔案清單）
            if n.endswith(".csv"):
                with z.open(n) as f:
                    yield pd.read_csv(                                # yield: 每次產出一個 DataFrame
                        f, header=None,                               # 因為原始 csv 沒有欄位名稱
                        names=["tradeId","price","qty","quoteQty",    # 自訂欄位名稱對應 Binance Trade 資料格式
                                "time","isBuyerMaker","isBestMatch"])

def backfill(start="2023-01-01", end=None):
    cur = date.fromisoformat(start)
    end = date.today() if end is None else date.fromisoformat(end)
    frames = []
    while cur <= end:
        z, ok = dl_one(cur)
        if z:
            for df in unzip_csvs(z):
                frames.append(df)
        cur += timedelta(days=1)
    
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.sort_values("time", inplace=True)
    out.to_csv("data/binance_trades/BTCUSDT_trades_2023on.csv", index=False)
    print("Saved", len(out), "rows")

backfill("2025-07-23")

# import asyncio, aiohttp, os
# from pathlib import Path
# from datetime import date, timedelta

# cfg = {
#     "base": "https://data.binance.vision",
#     "market": "futures/um",            # 現貨: spot；永續(USDT-M): futures/um
#     "symbol": "BTCUSDT",
#     "start": "2023-01-01",
#     "outdir": "data/binance_trades_zips",
#     "concurrency": 8,                  # 并發下載數
#     "timeout_s": 60,
# }

# # 目錄樣式：
# # monthly: data/<market>/monthly/trades/<SYMBOL>/<SYMBOL>-trades-2023-01.zip
# # daily:   data/<market>/daily/trades/<SYMBOL>/<SYMBOL>-trades-2023-01-01.zip


# def month_range(d0: date, d1: date):
#     cur = date(d0.year, d0.month, 1)
#     end = date(d1.year, d1.month, 1)
#     while cur <= end:
#         yield cur
#         # 先把他跨到當月的28號，再加4天 => 必定到下個月，再回到第一天 => 變成下個月的第一天開始繼續
#         cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

# def day_range(d0: date, d1: date):
#     cur = d0
#     while cur <= d1:
#         yield cur
#         cur += timedelta(days=1)


# def build_urls():
#     start = date.fromisoformat(cfg["start"])
#     today = date.today()
#     base = cfg["base"]; market = cfg["market"]; sym = cfg["symbol"]
#     # monthly first
#     mon = [f"{base}/data/{market}/monthly/trades/{sym}/{sym}-trades-{m.year}-{m.month:02d}.zip"
#            for m in month_range(start, today)]
#     # daily fallback
#     day = [f"{base}/data/{market}/daily/trades/{sym}/{sym}-trades-{d.isoformat()}.zip"
#            for d in day_range(start, today)]
#     return mon, day


# async def fetch_one(session, url: str, outdir: Path, sem: asyncio.Semaphore):
#     """
#     session	aiohttp 的 ClientSession: 共用連線池(瀏覽器)
#     url: str	欲下載的網址
#     outdir: Path	要儲存的資料夾（使用 pathlib.Path 更安全好用）
#     sem: asyncio.Semaphore:	限制同時進行的下載數量 (避免被伺服器ban)

#     await: 非同步等待，暫停這段、讓 event loop 去做別的事
#     """
#     fname = url.split("/")[-1]  # 取出檔名，例如 BTCUSDT-trades-2025-08-24.zip
#     dst = outdir / fname
#     if dst.exists(): return "skip"

#     async with sem:                 # 限制同時最多只能有 N 個下載在跑
#         for attempt in range(6):    # 嘗試最多 6 次
#             try:
#                 async with session.get(url, timeout=cfg["timeout_s"]) as r:
#                     if r.status == 200:
#                         data = await r.read()
#                         dst.parent.mkdir(parents=True, exist_ok=True)
#                         dst.write_bytes(data)
#                         return "ok"
#                     elif r.status == 404:
#                         return "404"
#                     else:
#                         await asyncio.sleep(1.5 * (attempt+1))
#             except Exception:
#                 await asyncio.sleep(2.0 * (attempt+1))
#     return "fail"




# async def main():
#     outdir = Path(cfg["outdir"]) / cfg["symbol"]; outdir.mkdir(parents=True, exist_ok=True)
#     mon_urls, day_urls = build_urls()           # 會產生兩組網址清單: 每月/每日合併的 zip
#     sem = asyncio.Semaphore(cfg["concurrency"]) # 控制同時最多幾個下載任務

#     # User-Agent: ok 防止某些伺服器拒絕匿名連線
#     async with aiohttp.ClientSession(headers={"User-Agent":"ok"}) as session:
#         # 先 monthly
#         res = await asyncio.gather(*[fetch_one(session, u, outdir, sem) for u in mon_urls]) # list，記錄每個 url 的下載狀態（"ok", "404", "fail", "skip"...）
        
#         # 對於 monthly 404 缺失的月份，用 daily 補回來
#         need_daily = []
#         for url, status in zip(mon_urls, res):
#             if status == "404":
#                 y_m = url.split("-trades-")[1].removesuffix(".zip")
#                 y, m = map(int, y_m.split("-"))
#                 # 當月所有日
#                 d0 = date(y, m, 1)
#                 d1 = (d0.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
#                 for d in day_range(d0, d1):
#                     need_daily.append(f"{cfg['base']}/data/{cfg['market']}/daily/trades/{cfg['symbol']}/{cfg['symbol']}-trades-{d.isoformat()}.zip")
#         if need_daily:
#             res_d = await asyncio.gather(*[fetch_one(session, u, outdir, sem) for u in need_daily])
#             print(f"daily done: ok={res_d.count('ok')}, skip={res_d.count('skip')}, 404={res_d.count('404')}, fail={res_d.count('fail')}")
#         print("monthly done: ok=", res.count("ok"), "skip=", res.count("skip"), "404=", res.count("404"), "fail=", res.count("fail"))

# if __name__ == "__main__":
#     asyncio.run(main())