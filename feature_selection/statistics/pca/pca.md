假設 feat 有 d 個，共 n 個時間點
=> 設 A 為一個 n*d 矩陣
=> 對 A 做中心化 + 標準化 => Z 

SVD: Z = UΣV^T, Σ: singular matrix
PC 分數（新特徵）: S = ZV = (UΣV^T)V = UΣ
⇒ 每一筆樣本在 PC𝑖 上的值 = 𝜎𝑖 × 該筆的左奇異座標 u_{t,i}

第 𝑖 個主成分的解釋變異比（EVR）： EVR_i = (σ_i)^2 / ∑(σ_j)^2
=> 累計到第 k 個, s.t. EVR >= 0.95 就停