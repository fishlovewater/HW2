# AIoT HW2 — Red Wine Quality Prediction with Linear Regression

> 以 **線性回歸 (Linear Regression)** 預測紅酒品質分數，完整遵循 **CRISP-DM** 流程，
> 包含 **特徵選擇 (Feature Selection)**、**模型評估 (Model Evaluation)**，
> 以及帶有 **信賴區間 / 預測區間** 的預測圖。

學號：`7114056032`

---

## 1. 資料集來源 (Dataset Source)

| 項目 | 內容 |
|------|------|
| 名稱 | Red Wine Quality (Cortez et al., 2009) |
| 來源 | Kaggle / UCI Machine Learning Repository |
| 連結 | https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009 |
| 原始論文 | P. Cortez et al., *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems, 2009 |
| 樣本數 | 1,599 筆（去除重複後 1,359 筆） |
| 特徵數 | **11 個**（符合作業「10–20 個特徵」要求） |
| 預測目標 | `quality`（感官品質分數 0–10，實際出現 3–8），以連續變數建模 |

**11 個特徵（皆為理化檢測數值）：**
fixed acidity、volatile acidity、citric acid、residual sugar、chlorides、
free sulfur dioxide、total sulfur dioxide、density、pH、sulphates、alcohol。

---

## 2. 專案結構

```
Aiot_hw2/
├── 7114056032_hw2.py          # 主程式（完整 CRISP-DM 流程）
├── README.md                  # 本文件
├── report.md                  # 報告（可轉存為 PDF）
├── 資料來源.txt                # 資料集連結
├── dataset/
│   └── winequality-red.csv    # 原始資料（本地備援）
├── results/                   # 執行後自動產生
│   ├── 01_target_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_feature_selection.png
│   ├── 04_model_comparison.png
│   ├── 05_residual_diagnostics.png
│   ├── 06_prediction_intervals.png     # ★ 帶信賴/預測區間的預測圖
│   ├── 07_actual_vs_predicted.png
│   └── metrics.csv
└── web/                       # ★ 互動式網站（可部署到 Vercel）
    ├── index.html
    ├── style.css
    ├── app.js                 # 以 Chart.js 讀取 data/results.json 繪圖
    ├── vercel.json
    ├── README.md              # 網站 / 部署說明
    ├── data/results.json      # 主程式產生的分析結果（自動）
    └── images/*.png           # 主程式複製進來的圖檔（自動）
```

---

## 3. 執行方式 (How to Run)

需要 Python 3.8+ 與下列套件：
`pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, statsmodels`

```bash
# 安裝套件（若使用 Anaconda 通常已內建）
pip install pandas numpy scikit-learn matplotlib seaborn scipy statsmodels

# 執行主程式
python 7114056032_hw2.py
```

> 本機開發環境使用 Anaconda 的 Python 直譯器
> （`C:\Users\user\anaconda3\python.exe`），已內建全部套件。
> 執行後所有圖表與指標會輸出到 `results/` 資料夾，終端機也會印出每個 CRISP-DM 階段的說明，
> 同時把網站所需資料寫入 `web/data/results.json` 並複製圖檔到 `web/images/`。

**資料取得（Data Acquisition）：** 主程式會先用 `requests` 從網路擷取 CSV
（仿 50-Startups 作法，檢查 HTTP 200），擷取失敗時自動改用本地
`dataset/winequality-red.csv`，確保一定能執行。資料網址：
`https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv`

---

## 4. CRISP-DM 流程 (方法論)

### 4.1 Business Understanding（商業理解）
目標：僅用 11 項客觀理化檢測，預測紅酒的感官品質分數。
若模型可靠，釀酒廠即可用便宜、客觀的實驗數據估計品質，
減少對昂貴且主觀的人工品評小組的依賴。本題屬 **監督式回歸** 問題。

### 4.2 Data Understanding（資料理解）
- 1,599 筆、12 欄（11 特徵 + 1 目標），**無缺失值**。
- 目標分佈集中在 5、6 分（典型不平衡），極端分數樣本少。
- 與品質相關性最高者：`alcohol` (r≈0.48)、`volatile acidity` (r≈-0.39)、`sulphates` (r≈0.25)。
- 產出圖：`01_target_distribution.png`、`02_correlation_heatmap.png`。

### 4.3 Data Preparation（資料準備）
- 移除 240 筆重複列 → 保留 1,359 筆。
- 以 80/20 切分訓練 / 測試集（`random_state=42`）。
- 使用 `StandardScaler` 標準化，**僅以訓練集 fit**，避免資料洩漏。

### 4.4 Feature Selection（特徵選擇）
使用三種方法並取「至少被 2 種方法選中」的共識特徵：
1. **SelectKBest**（f_regression 單變量 F 檢定）
2. **RFE**（以線性回歸遞迴消除特徵）
3. **LassoCV**（L1 正則化，係數歸零即剔除）

共識特徵（9 個）：fixed acidity、volatile acidity、citric acid、chlorides、
free sulfur dioxide、total sulfur dioxide、pH、sulphates、alcohol。
產出圖：`03_feature_selection.png`。

### 4.5 Modeling（建模）
共訓練 5 個模型：
| 模型 | 說明 |
|------|------|
| Simple LR (alcohol) | 單純線性回歸，僅用最強單一特徵 |
| Multiple LR (all) | 多元線性回歸，使用全部 11 特徵 |
| Multiple LR (selected) | 多元線性回歸，使用特徵選擇後的 9 特徵 |
| Ridge | L2 正則化（RidgeCV 自動選 alpha） |
| Lasso | L1 正則化（LassoCV 自動選 alpha） |

### 4.6 Evaluation（評估）
評估指標：**MAE、MSE、RMSE、R²、Adjusted R²** 以及 5-fold 交叉驗證 R²。
並檢驗線性回歸假設（殘差 vs 擬合值、Normal Q-Q）。

**測試集結果：**

| Model | n_feat | MAE | MSE | RMSE | R² | Adj_R² | CV_R² |
|-------|:---:|---:|---:|---:|---:|---:|---:|
| Simple LR (alcohol) | 1 | 0.5675 | 0.5368 | 0.7327 | 0.2421 | 0.2393 | 0.2171 |
| Multiple LR (all) | 11 | 0.5041 | 0.4310 | 0.6565 | 0.3915 | 0.3658 | 0.3209 |
| Multiple LR (selected) | 9 | 0.5032 | 0.4272 | 0.6536 | 0.3969 | 0.3762 | 0.3269 |
| Ridge | 11 | 0.5038 | 0.4276 | 0.6539 | 0.3964 | 0.3709 | 0.3223 |
| **Lasso** | 11 | **0.5015** | **0.4257** | **0.6525** | **0.3990** | **0.3736** | **0.3263** |

- **最佳模型：Lasso**（測試 R²≈0.399，RMSE≈0.653）。
- 從單純線性回歸 (R²=0.242) → 多元回歸 (R²≈0.40)，加入多特徵明顯提升。
- 特徵選擇後的 9 特徵模型 Adjusted R² 比全特徵更佳，代表剔除雜訊特徵有效。
- 產出圖：`04_model_comparison.png`、`05_residual_diagnostics.png`。

### 4.7 預測圖（含信賴 / 預測區間）
以 `statsmodels` OLS 擬合 `quality ~ alcohol`，繪製：
- 深藍回歸線
- **95% 信賴區間**（迴歸線平均值，窄）
- **95% 預測區間**（個別觀測值，寬）

產出圖：`06_prediction_intervals.png`、`07_actual_vs_predicted.png`。

### 4.8 Deployment（部署）
- 輸出標準化係數：`alcohol`(+0.31)、`volatile acidity`(-0.17)、`sulphates`(+0.14) 影響最大。
- 以 5 筆未見樣本示範預測。
- 實務上會將 `模型 + StandardScaler` 以 `joblib` 序列化，包成 API / 批次工作評分新酒款。

---

## 5. 結果圖表 (Results)

| 檔案 | 說明 |
|------|------|
| `01_target_distribution.png` | 品質分數分佈 |
| `02_correlation_heatmap.png` | 特徵相關係數熱力圖 |
| `03_feature_selection.png` | SelectKBest 特徵重要度 |
| `04_model_comparison.png` | 各模型 RMSE / R² 比較 |
| `05_residual_diagnostics.png` | 殘差診斷（線性/常態假設） |
| `06_prediction_intervals.png` | **預測圖 + 95% 信賴/預測區間** |
| `07_actual_vs_predicted.png` | 實際 vs 預測散佈圖 |

---

## 6. AI 協助與研究脈絡

- **GPT 輔助內容**：見 `report.md` 的「GPT 輔助內容」章節；完整對話需以 pdfCrowd 或其他方式匯出為 PDF，附於繳交報告。
- **NotebookLM 摘要**：見 `report.md` 的「NotebookLM 研究摘要」章節（100 字以上）。
- **主流 / 更優解法比較**：見 `report.md`。

---

## 7. Kaggle 名次 / 結果評估

此資料集為開放練習資料（非積分競賽），因此以 **預測圖 + 評估指標** 作為成果呈現：
最佳 Lasso 模型 R²≈0.40、RMSE≈0.65。文獻中線性模型於本資料集 R² 多落在 0.35–0.40，
本結果與主流一致；若改用隨機森林 / 梯度提升等非線性模型，R² 可提升至 0.45–0.55。

---

## 8. 互動式網站與 Vercel 部署

除了 Python 圖表，本專案另提供一個 **互動式網頁儀表板**（`web/`），
以 Chart.js 動態呈現 CRISP-DM 各階段結果，可一鍵部署到 **Vercel**。

### 8.1 內容
- KPI 卡片（資料筆數、特徵數、最佳模型、最佳 R²）
- 品質分佈長條圖、特徵相關性長條圖
- 特徵選擇 F-score + 共識特徵標籤
- 五模型 R² / RMSE 比較圖 + 完整指標表（最佳模型加亮）
- **Actual vs Predicted 散佈圖**（仿 50-Startups，含完美預測線）
- **OLS 預測圖 + 95% 信賴區間 + 95% 預測區間**
- 標準化係數圖、部署示範預測、Matplotlib 圖庫

### 8.2 本機預覽
```bash
python 7114056032_hw2.py          # 先產生 web/data/results.json
cd web
python -m http.server 8099        # 以 http:// 開啟（不可直接雙擊 html）
# 瀏覽器開 http://localhost:8099
```

### 8.3 部署到 Vercel
最簡單用 CLI：
```bash
npm i -g vercel
cd web
vercel          # Framework 選 Other；Root 為目前 web/ 目錄
vercel --prod   # 正式部署，取得公開網址
```
或用 GitHub 連動：到 vercel.com 匯入 repo，**Root Directory 設為 `web`**，
Framework 選 **Other**，Build/Output 留空即可。詳見 `web/README.md`。

> 網站為純靜態（HTML/CSS/JS + JSON），不需後端或 Python runtime，
> 因此在 Vercel 免費方案即可穩定運行、載入快速。

### 8.4 部署後請補上連結
部署完成後，將 Vercel 公開網址填入此處：

- 🌐 線上 Demo：`https://<your-project>.vercel.app`
- （選填）GitHub repo：`https://github.com/<you>/<repo>`
