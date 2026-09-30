# Phishing URL Detection

以 **1D-CNN、MLP 與 Intermediate Fusion（中間層融合）** 為核心的釣魚／惡意網址偵測系統。

> ☁️ **專案相關資料與模型檔案可直接存取**：[Google Drive 專案共用資料夾](https://drive.google.com/drive/folders/17T7RgNIsfq-1ENHAtQaOciVt52UlmGBP?usp=drive_link) 

使用方式: 
1. 請下載environment內的phishing_url_detection_system檔案，並打開https://colab.research.google.com/，將此檔案放入colab。
2. 請將code資料夾的所有程式放入colab記事本左側檔案的content資料夾內。
3. 可直接看輸出格的內容，也可以從頭執行一次。
---

## 📌 專題核心架構

```text
                         URL
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
           1D-CNN                    MLP
       字元層級特徵              URL 結構化特徵
              │                       │
              └───────────┬───────────┘
                          ▼
                 Intermediate Fusion
                    （特徵融合）
                          │
                          ▼
                     分類結果

📚 參考資料與文獻來源 (References & Datasets)
本專案在資料來源與相關研究上參考了以下公開資料集與學術文獻：

1. Malicious URLs Dataset (本專案核心訓練資料集)

資料集來源：Kaggle - Malicious URLs Dataset

說明：包含超過 65 萬筆整合後的多分類網址資料（Benign, Defacement, Phishing, Malware）。

2. UCI Website Phishing Dataset

資料集來源：Kaggle - Phishing Dataset UCI ML CSV（原始來自 UCI Machine Learning Repository: Website Phishing Dataset）

對應學術論文：Abdelhamid, N., Ayesh, A., & Thabtah, F. (2014). Phishing detection based associative classification data mining. Expert Systems with Applications, 41(13), 5948–5959. 

3. PhiUSIIL Phishing URL Dataset

資料集來源：Kaggle - PhiUSIIL Phishing URL Dataset

相關討論與來源：PhiUSIIL Discussion

對應學術論文：Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. Computers & Security, 103545. 