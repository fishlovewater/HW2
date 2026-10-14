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

本資料夾即為網站根目錄，不需要安裝前端依賴或執行建置。

### Vercel CLI

於專案根目錄執行：

```bash
npx vercel@latest login
cd web
npx vercel@latest --prod
```

第一次部署時選擇帳號／團隊並建立或連結專案，Framework 選 **Other**，目錄使用目前位置，不設定 Build Command。後續於 `web/` 執行 `npx vercel@latest --prod` 即可更新正式網站。詳見 [Vercel CLI 部署文件](https://vercel.com/docs/cli/deploy)。

### GitHub 整合

1. 在 Vercel 匯入 [fishlovewater/HW2](https://github.com/fishlovewater/HW2)。
2. **Root Directory** 設為 `web`。
3. Framework Preset 選 **Other**，Build Command 留空，Output Directory 使用預設值。
4. 部署完成後，將實際正式網址記錄於本文件及專案根目錄 README。

## 專案連結

- 原始碼：[fishlovewater/HW2](https://github.com/fishlovewater/HW2)
- 線上 Demo：[https://web-phi-ecru-34.vercel.app](https://web-phi-ecru-34.vercel.app)

## 部署驗證與更新

- 確認首頁、`/data/results.json` 及 `/images/01_target_distribution.png` 可讀取，圖表及模型指標正常顯示。
- 修改模型或資料後，先在專案根目錄執行 Python 主程式，再於 `web/` 重新部署。
- 網站載入預先產生的分析結果，瀏覽時不會執行 Python 或重新訓練模型。
- `vercel.json` 設定 clean URLs 與 JSON 快取重新驗證；`.vercel/` 專案連結資訊已被 `.gitignore` 排除。
- 若 `data/results.json` 不存在，請先執行主程式。
