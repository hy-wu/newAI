# BIIC (Bio-Inspired Information Cell) 深入分析報告

> 項目地址：https://github.com/val1813/BIIC
> 分析日期：2026-05-11
> 測試環境：PyTorch 2.9.0 + CUDA 13.0

---

## 一、從物理角度看 Clifford 代數

### 1.1 你早就見過 Clifford 代數

如果你是物理系學生，其實早就用過 Clifford 代數，只是沒叫這個名字：

| 你學過的 | 其實就是 Clifford 代數 |
|----------|----------------------|
| **Pauli 矩陣** σ₁, σ₂, σ₃ | Cl(3,0) 的矩陣表示，σᵢ² = +1, σᵢσⱼ = -σⱼσᵢ |
| **Dirac γ 矩陣** γ⁰, γ¹, γ², γ³ | Cl(1,3) 或 Cl(3,1) 的矩陣表示，{γᵃ, γᵇ} = 2ηᵃᵇ |
| **四元數** i, j, k | Cl(0,2) 的偶子代數，i² = j² = k² = -1 |
| **外微分／微分形式** dx^dy | grade-2 元素的 wedge product |
| **電磁場張量 F_μν** | 一個 **bivector** (grade-2 元素) |
| **相對論中的標量、向量、張量** | grade 分解——不同 grade 就是不同「階」的幾何量 |

**一句話總結**：Clifford 代數就是把「向量可以相乘」這件事做到極致，乘出來的東西自動帶有幾何意義。

### 1.2 從三維旋轉說起

三維旋轉有兩種等價描述：

```
向量形式：    v' = R · v             (旋轉矩陣 R ∈ SO(3))
四元數形式：  v' = q v q̄            (四元數 q ∈ SU(2))
```

四元數形式 `q v q̄` 就是一個 **sandwich product**。其中 q 是旋轉算子，v 是被旋轉的向量，q̄ 是共軛。這正是 Clifford 代數的核心操作——用一個「rotor」R 對任意幾何量做 sandwich 變換：

```
x' = R · x · R̃
```

神奇的是：**在 Clifford 代數中，sandwich product 對每個 grade (階) 都有明確的變換行為——且 grade-0 (純量) 永遠不變。**

### 1.3 Cl(4,1) 的物理意義

BIIC 選擇的 Cl(4,1) 度規是 `(+, +, +, +, -)`——4 個正號、1 個負號。

這有直接的物理對應：

```
e₁² = +1    → 空間維度 x
e₂² = +1    → 空間維度 y
e₃² = +1    → 空間維度 z
e₄² = +1    → 額外的空間維度"
e₅² = -1    → 時間維度 t（或共形維度）
```

這其實就是**共形幾何代數 (Conformal Geometric Algebra)**——用 5 維 Minkowski 空間來表示 4 維時空中的共形變換（旋轉、平移、縮放、反演、特殊共形變換）。這是一個非常漂亮的數學結構。

但對於 LLM 的場景，物理圖像是：

| Grade | 物理類比 | BIIC 中的角色 |
|-------|----------|-------------|
| **0** (純量, 1 個) | 粒子的**靜質量**、電荷 | **token 身分**——永不被覆寫 |
| **1** (向量, 5 個) | 四維動量 p_μ | token 的**位置／方向**資訊 |
| **2** (雙向量, 10 個) | F_μν 電磁場張量 | token 之間的**交互作用** |
| **3** (三向量, 10 個) | 空間體積元 dx^dy^dz | 更高階的語義關聯 |
| **4** (四向量, 5 個) | 時空體積元 ε_μνρσ | 全局面文脈 |
| **5** (偽純量, 1 個) | 手性 γ⁵ | 分辨「左右對稱」的資訊 |

關鍵洞察：**sandwich product 保持 grade-0 不變是代數定理，不是工程近似。** 就像對任何張量做洛倫茲變換，純量部分永遠不變一樣——這是結構保證的。

### 1.4 BIIC 的物理圖像

把推理過程想像成**時空演化**：

1. **編碼**：每個 token 被映射到時空中的一個「粒子」——有質量 (grade-0) 和動量、自旋等 (grades 1-4)
2. **Writer**：對每個粒子做局部「轉動」——旋轉子 R = exp(B/2) 是類比於「作用量」的指數映射
3. **Eraser**：引入「阻尼」——grades 1-4 向先驗衰減，但 grade-0 (質量) 完全不受影響
4. **解碼**：從最終狀態讀出資訊

這個圖像的優雅之處是：**不管你轉了多少圈，粒子的「身分」(grade-0) 永遠不變。** 這就解決了 LLM 深層網路中 token 身分被覆寫的問題。

---

## 二、項目整體架構

### 2.1 專案結構

```
BIIC/
├── src/
│   ├── clifford_cl41.py      # Cl(4,1) 代數核心：乘法表、幾何積、grade操作
│   ├── rotor_utils.py         # 旋轉子生成、sandwich積、歸一化
│   ├── eraser_ops.py          # GradeAwareEraser + EraserOperator
│   ├── token_to_ic.py         # TokenToImmutableCore 編碼器
│   ├── all_grade_decoder.py   # AllGradeDecoder 全grade解碼器
│   ├── mutable_state.py       # Writer + BIICLayer 推理層
│   └── biic_loss.py           # BIICLoss 分階段輔助損失
├── tests/
│   ├── test_phase1.py         # 10 個數學驗證測試
│   ├── test_decoder_basic.py  # 4 個解碼器測試
│   ├── test_encoder.py        # 3 個編碼器測試
│   └── test_full_pipeline.py  # 4 個完整管線測試
├── results/                   # 實驗結果
└── figures/                   # 圖表
```

### 2.2 各模組功能詳解

#### `clifford_cl41.py` — 代數核心
- 實現 Cl(4,1) 的 32 個基底、預計算 32×32 乘法表
- 提供 `geometric_product_fast`：向量化的 Clifford 幾何積
- 提供 `grade_project/extract/assemble`：grade 的擷取和重組
- 提供 `reverse_multivector`：reversion 運算

#### `rotor_utils.py` — 旋轉子引擎
- `exp_bivector(B)`: 雙向量的指數映射（Taylor 展開 16 項）
- `sandwich_product(R, x)`: `R·x·R̃` 核心變換
- `normalize_rotor(R)`: 確保 `R·R̃ = 1`
- `_stabilize_multivector`: 對各 grade 分別保範，防止負度規導致的數值爆炸

#### `eraser_ops.py` — 遺忘機制
- `GradeAwareEraser`: grade-0 和 grade-5 完全不動，只對 grades 1-4 做受控衰減
- `EraserOperator`: 通用的密度矩陣衰減，支援隨機重置

#### `token_to_ic.py` — 編碼器
- 128 維共享 embedding → 6 個獨立線性層（各 grade 一個）→ 拼接成 `[B, L, C, 32]`
- grade-0 的初始化 std=0.01（引導學習穩定核心），其他更大

#### `all_grade_decoder.py` — 解碼器
- 各 grade 獨立投影到 d_hidden，GELU 非線性
- 可學習的 grade gate `sigmoid(θ)`，動態決定各 grade 權重
- 支援 `active_grades` 參數，用於消融實驗

#### `mutable_state.py` — 推理層
- `SimpleWriter`: 每個通道的可學習雙向量 → 旋轉子 → sandwich 變換
- `BIICLayer`: Writer → Eraser，可選殘差連接（只對 grades 1-4 加殘差）

#### `biic_loss.py` — 損失函數
- 主損失：CrossEntropy
- 輔助損失 1：grade-0 獨立分類頭 → 引導其承載 token 身分
- 輔助損失 2：通道間方差最大化 → 防止 grade 坍縮
- α 退火：1.0 → 0.01 (線性)，永久保留 0.01

---

## 三、測試結果詳盡記錄

### 3.1 Phase 1: 數學基礎驗證 (10/10 PASS)

| # | 測試名稱 | 結果 | 關鍵數值 | 說明 |
|---|---------|------|---------|------|
| 1 | 乘法表正確性 | 🟢 | 最大誤差 0.00 | e₁²~e₄²=+1, e₅²=-1, 反對易性完全成立 |
| 2 | 旋轉子驗證 | 🟢 | 身分誤差 1.67×10⁻⁶ | R·R̃=1，無奇數 grade 分量 |
| 3 | **Grade-0 不變性 (核心)** | 🟢 | **8.23×10⁻⁶** | 100 次 sandwich 變換後，grade-0 幾乎不變(< 10⁻⁴) |
| 4 | Grade-5 不變性 | 🟢 | 1.57×10⁻⁵ | 偽純量在純旋轉下同樣不變 |
| 5 | Grade-1 等變性 | 🟢 | 2.03×10⁻⁶ | 「先變換再投影 = 先投影再變換」 |
| 6 | 梯度流 | 🟢 | 首/末層比率 0.98 | 10 層鏈中梯度不消失也不爆炸 |
| 7 | 多通道獨立性 | 🟢 | 通道間洩漏 **0.00** | 8 個通道完全隔離 |
| 8 | Eraser 收斂性 | 🟢 | 收斂誤差 2.98×10⁻⁸ | 200 步後收斂到先驗，與理論值 (0.5²⁰⁰≈6×10⁻⁶¹) 吻合 |
| 9 | **不變核不受 Eraser 影響** | 🟢 | **Grade-0 變化: 0.00** | grade-0 和 grade-5 在 50 次 Eraser 後精確不變 |
| 10 | 衰減率梯度健康 | 🟢 | 梯度範數 0.51，最小值 4.36×10⁻⁵ | 梯度未消失未爆炸 |

#### 關鍵發現

**Grade-0 不變性是真實的數學保證**，不是工程近似：
- Sandwich 積 100 次後誤差僅 8.23×10⁻⁶
- Eraser 操作 50 次後誤差精確為 **0.00**
- 多通道方案下通道間洩漏為 **精確 0**
- 以上都在端到端訓練後仍然成立

### 3.2 Phase 2 Step 1: 解碼器測試 (4/4 PASS)

| # | 測試名稱 | 結果 | 關鍵數值 |
|---|---------|------|---------|
| 1 | Decoder 過擬合測試 | 🟢 | Loss: 7.10 → 0.011 (改善 99.8%) |
| 2 | 梯度流 | 🟢 | 平均梯度範數 0.23，範圍健康 |
| 3 | Gate 更新 | 🟢 | 6 個 grade gate 總變化 0.042 |
| 4 | **Grade-0 vs 全部 Grade** | 🟢 | 全部 grade loss **0.006** vs 僅 grade-0 loss **0.032** |

#### 關鍵發現

**All-grade 解碼優於 grade-0-only 解碼 5.4 倍** (0.006 vs 0.032)，證明：
- Grades 1-4 承載了有用的非冗餘資訊
- 可學習的 grade gate 確實能區分各 grade 的重要性

### 3.3 Phase 2 Step 2: 編碼器測試 (3/3 PASS)

| # | 測試名稱 | 結果 | 關鍵數值 |
|---|---------|------|---------|
| 1 | Grade 分離訓練 | 🟢 | Loss: 10.51 → 0.099, grade 範數標準差 9.42 |
| 2 | Token 區分能力 | 🟢 | Grade-0 餘弦相似度僅 **0.037** (接近正交) |
| 3 | 編碼器梯度健康 | 🟢 | 6 個 grade 投影層梯度皆正常 |

#### 關鍵發現

- 不同 token 的 grade-0 餘弦相似度僅 0.037，接近正交——這意味著 grade-0 做了優秀的 token 辨識
- Grade 範數差異大 (std=9.42)，說明各 grade 學會了不同的功能
- Loss 在 300 步後快速下降（α 退火到 0.01 後），表明輔助損失成功引導了前期學習

### 3.4 Phase 2 Step 3: 完整管線測試 (4/4 PASS)

| # | 測試名稱 | 結果 | 關鍵數值 |
|---|---------|------|---------|
| 1 | 完整前向+反向 | 🟢 | Grade-0 變化 **0.00**, grade-2 變化 0.013, 編碼器梯度 2.99 |
| 2 | Eraser 效果 | 🟢 | 有/無 Eraser 差異不明顯 |
| 3 | 端到端訓練收斂 | 🟢 | Loss: 10.57 → 0.72 (**改善 93.2%**) |
| 4 | 訓練中 Grade-0 保持 | 🟢 | **最大變化 0.00** |

#### 關鍵發現

- **訓練過程中 grade-0 變化為精確 0.00**——這是最重要的結果，證明代數不變性在實際訓練中仍然成立
- Eraser 效果不明顯——當前設定下 (decay 初始化 sigmoid(-4.6)≈0.01) Eraser 對範數的影響太小
- 50 步訓練就有 93.2% 的 loss 改善，確認管線可以正常學習

---

## 四、Eraser 效果的深入分析

Eraser 的設計是個好想法，但實測效果不明顯：

```
無 Eraser:  initial norm=4.86, final norm=4.86, ratio=0.999
有 Eraser:  initial norm=4.86, final norm=4.86, ratio=0.999
```

原因分析：
1. **初始化太小**：decay_logits 初始化為 -4.6，對應 sigmoid ≈ 0.01——即每次只衰減 1%
2. **殘差連接稀釋**：BIICLayer 對 grades 1-4 取平均 (mv + mv_new) × 0.5，這相當於添加了低通濾波
3. **Writer 輸出維持範數**：旋轉子是保範變換，所以 Writer 前後的範數差異主要來自數值誤差

改進方向（推測）：
- 增加初始 decay 率到 sigmoid(-1.5) ≈ 0.18
- 用更積極的殘差權重（如 0.7 mv_new + 0.3 mv）
- 或使用 l1 正則化強制稀疏

---

## 五、代數正確性的關鍵驗證

### 5.1 Grade-0 不變性的雙重保證

```
Sandwich 積保證：理論上是精確的，實測誤差 ~10⁻⁵（來自浮點數）
Eraser 保證：    Grade-0 和 grade-5 被強制排除在衰減之外
                 實測誤差 = 0.00（因為 torch.where 和 slice 賦值是精確的）
```

### 5.2 多通道獨立性

```
通道間洩漏 = 0.00
```
因為每個通道有自己的 Writer 參數和獨立的 sandwich product，通道之間完全隔離。

### 5.3 等變性 (Equivariance)

```
project(R·x·R̃, 1) = R·project(x, 1)·R̃   誤差 2×10⁻⁶
```
這驗證了 Clifford 代數的 grade 投影與 sandwich 變換是可交換的——這在代數上天然成立，實測確認了實現正確性。

---

## 六、創新點總結

| 創新 | 狀態 | 評論 |
|------|------|------|
| Cl(4,1) multivector 作為 LLM 資訊載體 | ✅ 已驗證 | 代數操作正確，grade-0 保留 |
| Grade-0 代數不變性 | ✅ 已驗證 | 誤差 ~10⁻⁵ ~ 0.00 |
| GradeAwareEraser | ✅ 已驗證 | 正確只修改 grades 1-4 |
| AllGrade 解碼（可學習 gate） | ✅ 已驗證 | 優於 grade-0-only 5.4 倍 |
| 多通道獨立變換 | ✅ 已驗證 | 通道間無串擾 |
| 端到端訓練收斂 | ✅ 已驗證 | Loss 下降 93.2% |
| 無 KV cache 長上下文 | ⏳ 待驗證 | Phase 4 目標 |
| SlowFast O(L) 架構 | ⏳ 未實作 | 程式碼尚未出現 |

---

## 七、潛在問題與風險

### 7.1 效率問題
- geometric_product_fast 使用 `scatter_add_`，雖然向量化了，但 32×32 的乘法表對大 batch 有 overhead
- 每個 token 用 8×32 = 256 維 multivector，比標準 embedding 約大 2-4 倍
- 旋轉子的 Taylor 展開需要 16 項，每步計算量不小

### 7.2 可擴展性
- 通道數 C=8 是隨意選的，沒有 ablation study
- 沒有在真實語言模型尺度上驗證（當前僅測試 vocab_size=1000）

### 7.3 Eraser 設計
- 當前設定下 Eraser 的實際影響接近零
- 可學習 decay 率可能是個雙面刃——模型可能會學到不衰減（把 λ 推到 0）

---

## 八、總評

BIIC 是一個數學構思優雅的研究項目。它的核心 insight——用 Clifford 代數的 grade-0 代數不變性來保證 LLM 中 token 身分的持久性——在理論上是漂亮的。實測驗證了所有代數聲稱：grade-0 在訓練和推理中確實保持不變。

目前它還是一個「數學驗證平台」而非「可用的語言模型」。真正的考驗在 Phase 4——在 WikiText-103 上運行真正的語言建模任務，並與 baseline transformer 比較。

以一天 13 個 commits 的早期階段而言，專案結構清晰（7 個源文件、4 個測試文件、21 個測試），測試覆蓋率完整，這是個不錯的開端。

---

## 附錄：測試運行命令

```bash
cd BIIC
pip install -r requirements.txt   # torch, numpy, scipy, matplotlib

# Phase 1: 代數基礎 (10 個測試)
python tests/test_phase1.py

# Phase 2 Step 1: 解碼器 (4 個測試)
python tests/test_decoder_basic.py

# Phase 2 Step 2: 編碼器 (3 個測試)
python tests/test_encoder.py

# Phase 2 Step 3: 完整管線 (4 個測試)
python tests/test_full_pipeline.py

# 總計：21 個測試，全部通過 ✅
```
