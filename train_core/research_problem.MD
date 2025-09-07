1.  目前用的是time bars，本身統計性質差 (非恆定變異 (heteroscedasticity)，違反統計學假設)
=> 用 event-driven bars：
    Tick/Volume/Dollar Imbalance Bars (TIB/VIB/DIB)
    Runs Bars (TRB/VRB/DRB)
    CUSUM filter events

    再把 event-level 特徵 aggregate 回 15m 視窗

2. 分類 (cls) 與 回歸 (reg) 混用
    (目前用 regression（預測未來報酬值），再用 threshold 決定方向)
=> 拆開任務 → Multi-task 或 Meta-Labeling

3. 篩選feature應更多考慮4.4 Average Uniqueness of a Label
    => BAGGING CLASSIFIERS AND UNIQUENESS
    以及time decay

4. Fractionally Differentiated Features
    考慮是否不只用logret作為target?
    => 用Fixed-Width Window Fracdiff, FFD (固定視窗分數差分)
    ADF 單位根檢定 (Augmented Dickey-Fuller test)

5. model: RF = Bagging + 特徵隨機性。
   金融數據因冗餘度高 → 必須搭配 avgU + Sequential Bootstrap 才能真正發揮效果。

6. FEATURE IMPORTANCE 8.4.3 加權 Kendall’s Tau
    => 用MDA/MDI檢測，然後做PCA，檢查兩者是否相關? 
    有相關表示前幾高的是真的有用