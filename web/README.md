# 紅酒品質 MLR 儀表板（Vercel 靜態網站）

這是一個純前端的互動式儀表板，視覺化紅酒品質多元線性回歸的 CRISP-DM 全流程結果。
所有數據由主程式 `../7114056032_hw2.py` 產出並寫入 `data/results.json`，
網頁以 Chart.js 讀取並繪圖，因此部署到 Vercel 時 **不需要後端或 Python runtime**，速度快又免費。

## 目錄結構

```
web/
├── index.html        # 頁面結構
├── style.css         # 樣式
├── app.js            # 讀取 data/results.json 並以 Chart.js 繪圖
├── vercel.json       # Vercel 設定（cleanUrls + 快取標頭）
├── data/
│   └── results.json  # 由主程式產生的分析結果（指標、座標、區間…）
└── images/           # 由主程式複製進來的 Matplotlib 圖檔
```

## 本機預覽

```bash
# 於 web/ 目錄啟動簡易伺服器（需透過 http:// 開啟，不能直接雙擊 index.html，fetch 會被瀏覽器擋）
python -m http.server 8099
# 瀏覽器開啟 http://localhost:8099
```

## 重新產生資料

修改模型或資料後，於專案根目錄重新執行主程式即可更新網頁資料：

```bash
python 7114056032_hw2.py
# 會重新寫入 web/data/results.json 與 web/images/*.png
```

## 部署到 Vercel

本資料夾即為網站根目錄。三種方式擇一：

### 方法 A：Vercel CLI（最快）
```bash
npm i -g vercel
cd web
vercel          # 第一次會問設定，Framework 選 "Other"，Root 使用目前目錄
vercel --prod   # 正式部署
```

### 方法 B：GitHub + Vercel 網站
1. 將整個專案推上 GitHub。
2. 到 <https://vercel.com> → New Project → 匯入該 repo。
3. **Root Directory** 設為 `web`。
4. Framework Preset 選 **Other**（這是純靜態網站，無需 build）。
5. Build Command 留空，Output Directory 留空（直接服務 `web/`）。
6. Deploy，完成後即取得公開網址。

### 方法 C：直接拖拉
到 Vercel 後台用 "Deploy" 把 `web/` 資料夾拖進去即可。

## 說明

- 若 `data/results.json` 不存在，頁面會顯示提示，請先跑一次主程式。
- `app.js` 以相對路徑 `./data/...`、`./images/...` 讀檔，確保在 Vercel 子路徑下也能運作。
