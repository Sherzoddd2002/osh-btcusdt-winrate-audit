# OSH / BTCUSDT 15m — Diagnostika va yo'l xaritasi

> `OSH MAIN.ipynb` (31 cell, 26 tasida saqlangan output) asosida tayyorlandi.
> Hamroh fayl: `pipeline_v2.py` — ishga tushirib ko'rish mumkin bo'lgan tuzatilgan pipeline.
> **Bu investitsiya maslahati emas.** Hech bir model foyda kafolatlamaydi; kripto savdosi yuqori xavfli.

---

## 0. To'g'ridan-to'g'ri javob

**Aybdor: model emas.** Taxminiy ulush:

| Qism | Ulush | Izoh |
|---|---|---|
| Feature / target / ma'lumot manbai | ~70% | Eng katta muammo shu yerda |
| Baholash dizayni va iqtisodiy xarajatlar | ~25% | "Yaxshi natija" aslida yo'q |
| Model arxitekturasi (LightGBM) | ~5% | LightGBM bu yerda to'g'ri tanlov |

LightGBM'ni XGBoost / CatBoost / LSTM / Transformer'ga almashtirish ROC AUC'ni `0.514 → 0.520` atrofida siljitishi mumkin, xolos. Sabab oddiy: **BTC 15 daqiqalik yo'nalishi deyarli samarali bozor**, bitta shamdagi OHLCV + taker volume axboroti narxga allaqachon singib bo'lgan. 239 276 qator "katta baza" emas — bu 239 276 ta bir-biriga bog'liq, 1 soatlik targetda 4 tadan ustma-ust tushadigan kuzatuv (effektiv mustaqil namuna ~60 ming), ustiga 2020–2026 oralig'ida 4–5 xil bozor rejimi.

**Siz allaqachon to'g'ri qilgan narsalar** (bularni saqlang, ko'pchilik buni ham qilmaydi): xronologik 70/15/15 split, 8 shamlik purge, kalibratsiyadan muzlatilgan threshold, ustma-ust tushmaydigan savdolar, xarajatlar stress-testi, `model.joblib` + `features.json` + SHA-256 + `live_orders_enabled: False` + `SHADOW_MODE_ONLY`. Muammo muhandislik intizomida emas, **axborot va iqtisodiyotda**.

---

## 1. Notebook'dagi aniq raqamlar

| Ko'rsatkich | Validation (2024-10 → 2025-10) | Holdout (2025-10 → 2026-10) |
|---|---|---|
| ROC AUC (1H) | 0.5139 – 0.5157 | — (o'lchanmagan) |
| PR AUC (1H) | 0.5110 – 0.5162 | — |
| **Brier (1H)** | **0.2603 – 0.2609** | — |
| TOP-5% precision (15m, C model) | **56.35%** | **52.84%** |
| TOP-5% o'rtacha return (15m, C) | +1.29 bps | **−0.80 bps** |
| B_TOP10 gross / savdo | +2.65 bps (554 ta) | +0.55 bps (3310 ta) |
| B_TOP5 gross / savdo | +1.36 bps (319 ta) | **+1.38 bps (2236 ta)** |
| TOP-5% threshold'dan o'tganlar ulushi | 5% (kutilgan) | **12.35%** ← kalibratsiya siljishi |
| 10 bps xarajatda B_TOP5 | — | **−8.62 bps/savdo, −85.8% compounded** |

**Eng og'riqli fakt:** `Brier = 0.2609`, doimiy `0.5` bashoratniki `≈ 0.2499`. Ya'ni model ehtimollari kvadratik xatolik bo'yicha **tanga tashlashdan ham yomon**. Boshqa tomondan `ROC AUC = 0.515` statistika jihatdan ahamiyatli bo'lishi mumkin (35 888 namunada SE ≈ 0.005), lekin iqtisodiy jihatdan ahamiyatsiz: 1.38 bps yalpi daromad, real round-trip xarajat 8–20 bps.

**Xarajatlar haqida aniq hisob (cell 25, B_TOP5, holdout):**

| Round-trip cost | Net / savdo | Compounded |
|---|---|---|
| 0 bps | +1.38 bps | +32.8% |
| 0.5 bps | +0.88 bps | +18.8% |
| 1 bps | +0.38 bps | +6.2% |
| 2 bps | −0.62 bps | −15.1% |
| **10 bps** | **−8.62 bps** | **−85.8%** |

Binance USDⓈ-M futures taker `0.04%` (maker `0.02%`) = 4 bps **har tomonga** → 8 bps RT; spot `0.1% / 0.1%` = 10 bps har tomonga → 20 bps RT (2026 tariflari; BNB bilan to'lash / VIP darajasi buni biroz kamaytiradi — o'z hisobingizdagi haqiqiy rate'ni API'dan oling). Bunga spread/slippage 1–5 bps va (1 soat ushlab turganda) funding qo'shiladi. **Sizning asosiy stsenariylaringiz 0–2 bps edi — bu real emas.** `10 bps` ssenariy eng optimistik real chegara, va unda barcha modellar zarar ko'rsatgan.

---

## 2. Topilgan 13 ta muammo

| # | Muammo | Dalil | Tuzatish |
|---|---|---|---|
| 1 | **Absolyut narx/volume feature'lari nostatsionar** | `open/high/low/close` gain ≈ 8%, `volume/quote_volume/trade_count/taker_*/sell_*` gain ≈ 18% (cell 14 top-20). BTC 2020 ≈ $7k, 2026 ≈ $81k. Daraxtning `close < 30000` spliti 2026'da ma'nosiz | Faqat `log return`, nisbat, rolling z-score |
| 2 | **Bitta sham konteksti, hech qanday lag/rolling yo'q** | 42 feature'ning barchasi joriy shamdan | 1/2/4/8/16/32/96 shamlik return, ATR/vol, VWAP/SMA'dan og'ish, oqim imbalance'ning rolling o'rtachasi |
| 3 | **Deterministik dublikatlar** | `buy_ratio ≈ buy_quote_ratio ≈ 1 − sell_ratio`; `close_vol ≈ open_vol ≈ volume`; `yo'nalish = 2·buy_volume − volume`; `sell_quote_volume = quote_volume − buy_quote_volume`. Cell 5'da korrelyatsiya hisoblangan, lekin hech narsa o'chirilmagan | Har guruhdan bittasini qoldirish |
| 4 | **Vaqt feature'lari shovqin** | `mod_sin + mod_cos` = 7.24% gain, ustiga `hour/dow/month_sin/cos` — 13 ta ustun. 6.8 yil ichida tasodifiy davriylikni yodlab oladi | Faqat Asia/EU/US sessiya one-hot + `is_weekend`; qolganini o'chirish |
| 5 | **Target noto'g'ri savol** | "Keyingi sham yashilmi?" → UP ulushi 50.19%, drift ≈ 0, fat-tail, SNR juda past. Brier > 0.25 | Triple-barrier / vol-normallashtirilgan / xarajatdan keyingi target, meta-labeling |
| 6 | **Pattern yo'nalishi teskari** | `BULLISH_FLOW` (siz LONG) → keyingi UP **46.33%**; `UP_LOWER_REJECTION` (LONG) → **46.48%**; `DOWN_UPPER_REJECTION` (siz SHORT) → **51.16%**; `NO_PATTERN` → **52.05%**. Cell 9'da deyarli barcha gross mean manfiy | Bu qisqa muddatli **reversal**. Yo'nalishni invert qiling yoki pattern'larni butunlay tashlab, modelga o'rgating |
| 7 | **Eng barqaror "pattern" — pattern yo'qligi** | Cell 8: `NO_PATTERN` benchmarkdan 76.9–84.6% oylarda yaxshi; `BULLISH_FLOW` atigi 7.7–30.8% | Filtrlar anti-selektiv ekanini tan olish |
| 8 | **Patternlar butun datasetda qo'lda o'ylab topilgan** | `0.50`, `1.2`, `0.65` konstantalari 239k qator ko'rib tanlangan | Data-snooping; barcha gipotezalarni faqat train'da yaratish |
| 9 | **Kalibratsiya siljishi** | Validation TOP-5% threshold `0.53006` → holdoutda **12.35%** signal o'tgan (5% kutilgan). Ehtimollar rejim bilan ko'chadi | Rolling rank/kvantil yoki isotonic/Platt kalibratsiya; `prob_up`ni ehtimol sifatida ishlatmang |
| 10 | **Holdout ifloslangan** | Bir davr cell 16, 17, 23, 25'da 4 marta ochilgan; keyin `B_MORPHOLOGY_TOP5` aynan shu holdout natijasiga qarab tanlangan (cell 26/27) | Oxirgi 6–12 oy butunlay tegilmagan + forward paper-trading |
| 11 | **Ko'p sinov tarafkashligi** | 3 model × 3 horizon × 2 selection × 6 cost ≈ 100+ konfiguratsiya, bitta davrda | Deflated Sharpe Ratio / Bonferroni; har bir tajribani registrga yozish |
| 12 | **Reproducibility + xavfsizlik** | Cell 13: "Parametrlar manbasi: **Mavjud LightGBM model parametrlari**" (avvalgi sessiyadan qolgan obyekt). Cell 1'da DB paroli ochiq matn. Cell 2'da yalang'och `except:` | Params'ni aniq yozish + seed + versiya; parolni `.env`/env var'ga; `except Exception as e` |
| 13 | **Ma'lumot haqiqiyligi tekshirilmagan** | 6.8 yilda **0 ta** interval uzilishi, 46% li barqaror reversal | Binance API'dan yangi yuklab solishtiring: dublikat, zero-volume sham, UTC, `open_time` ms, `ignore_value` |

Kichik, lekin muhim: `train_test_split` import qilingan (hozircha ishlatilmagan) — vaqt qatorida tasodifiy split ishlatilsa, kelajak sizib o'tadi. Uni butunlay olib tashlang.

---

## 3. Nima uchun "ko'p ma'lumot" yordam bermayapti

1. **Avtokorrelyasiya:** 1H target 4 ta ketma-ket shamni qoplaydi → 239k qator ≈ 60k mustaqil namuna.
2. **Rejim siljishi:** 2020 (COVID), 2021 (bull), 2022 (bear/FTX), 2023–24 (ETF), 2025–26 — bir xil taqsimot yo'q. Eski ma'lumot yangi rejimga zarar yetkazishi mumkin. Yaqin davrga eksponensial og'irlik bering.
3. **Axborot zichligi:** 15m OHLCV — eng ommaviy, eng ko'p "chaynalgan" ma'lumot. Hajm oshirish yangi axborot qo'shmaydi.
4. **Signal emas, shovqin o'lchanyapti:** `Brier > 0.25` modelning ehtimollari bazaviy chastotadan uzoqlashib, o'ziga ishongan holda xato qilayotganini ko'rsatadi.

---

## 4. Feature retsepti (faqat stasionar)

Barchasi `t` shami **yopilganda** ma'lum. `pipeline_v2.py::make_features()`da tayyor.

| Guruh | Feature'lar |
|---|---|
| Sham shakli (nisbiy) | `log(close/open)`, `(high−low)/close`, `(close−low)/(high−low)`, `upper_wick/close`, `lower_wick/close`, `\|close−open\|/(high−low)` |
| Oqim imbalance | `2·(taker_buy_base/volume) − 1`, uning 96 shamlik z-score'i, quote-volume versiyasi |
| Momentum / reversal | `log(C_t/C_{t−k})`, `k ∈ {1,2,4,8,16,32,96}` |
| Volatillik / rejim | `std(ret)` 16/96/384, `vol_k / vol_k.rolling(384).mean()`, `range_rel` z-score, `ret.skew(96)`, lag-1 avtokorrelyasiya(96) |
| Narx konteksti | `C/SMA96 − 1`, `C/SMA384 − 1` |
| Faollik anomalisi | `volume`, `trade_count`, `quote_volume` ning 96 shamlik z-score'lari va log o'zgarishi |
| Vaqt | faqat `session_asia/eu/us`, `is_weekend` (UTC!) |
| **O'chiriladi** | barcha absolyut narxlar, absolyut volumelar, `month_*`, `dow_*`, `hour_*`, `mod_*`, barcha dublikatlar |

### Tashqi ma'lumot — haqiqiy edge shu yerda bo'lishi mumkin

Binance public API'dan olinadi, kline bilan `open_time` bo'yicha birlashtiriladi:

- `fapi/v1/fundingRate` (8 soatlik) + keyingi funding'gacha qolgan vaqt
- `fapi/v1/openInterest` va uning 1h/4h o'zgarishi
- `futures/data/takerlongshortRatio`, `topLongShortAccountRatio`, `topLongShortPositionRatio`
- `!forceOrders` WebSocket → likvidatsiyalar hajmi (ayniqsa rejim o'zgarishida)
- perpetual − spot basis, `ETH/BTC`, `DXY`, USDT dominansligi
- order book imbalance (agar `depth` stream'ni yozib borsangiz)

Bularsiz faqat OHLCV bilan "yutuqli" yo'nalish modeli kutish real emas.

---

## 5. Target retsepti

Hozirgi: `1 if next_close > next_open else 0` — momentumni emas, deyarli nolga teng driftni bashorat qilyapsiz.

**Tavsiya etiladigan zanjir:**

1. **Triple-barrier (López de Prado):** `σ_t = std(log ret, 96) · √h` — faqat `t` gacha bo'lgan ma'lumot. Yuqori to'siq `+k·σ`, pastki `−k·σ`, muddat `h` sham. Label `+1 / −1 / 0(timeout)`. `+1` faqat `k·σ > xarajat` bo'lganda → past volatillik davrlari avtomatik "savdo yo'q" bo'ladi.
2. **Xarajatni target ichiga kiriting:** `y = 1` faqat *round-trip'dan keyin* foydali bo'lsa. Shunda model `0 → 1` emas, `0 → 10 bps` chegarani o'rganadi.
3. **Meta-labeling:** avval oddiy qoida (masalan, siz topgan reversal) nomzod savdolarni chiqaradi; ML faqat "bu nomzod foydalimi?" deb javob beradi. Butun 239k qatorga emas, faqat 5–10% nomzodga o'rgatish → ancha yuqori SNR.
4. **Sample weight:** ustma-ust tushish uchun `1/h`, katta harakat uchun `clip(|r|/σ, 0.25, 3)`.
5. **Muqobil (ko'pincha osonroq) maqsad:** yo'nalish emas, **volatillik / rejim / likvidlik**ni bashorat qilish. U ancha bashoratli, va uni position sizing hamda "savdo qilmaslik" filtri sifatida ishlatish mumkin.

> `pipeline_v2.py` sintetik (hech qanday edge yo'q) ma'lumotda `AUC ≈ 0.58` chiqaradi, lekin `net = −10.5 bps`, `0/4 fold musbat`, **VERDICT: NO-GO**. Bu juda muhim dars: AUC "yaxshi" ko'rinishi mumkin (u qisman volatillikni bashorat qiladi), lekin strategiya baribir zarar qiladi. **Har doim net PnL'ga qarang.**

---

## 6. Baholash protokoli

```
[train 1 yil] [embargo 6] [test 90 kun] → [train 1y+90k] [embargo] [test 90 kun] → ...
```

- **Purged walk-forward**: kamida 8–12 fold, har biri xronologik, expanding train.
- Har fold'da: threshold faqat train oxiridan (kalibratsiya), test'ga tegilmaydi.
- Har fold'da: `AUC`, `Brier` (doimiy baseline bilan), `precision@top5`, `mean net bps`, `per-trade Sharpe`, `max drawdown`, benchmark (har `h` shamda long) bilan farq.
- **Yakuniy qaror bitta sondan emas, taqsimotdan:** median net > 0, musbat fold'lar ≥ 75%, eng yomon fold qabul qilinadigan, Deflated Sharpe (ko'p sinov uchun tuzatma) ahamiyatli.
- **Ishonch oralig'i:** 2 236 savdoda 52.37% precision → SE ≈ 1.06%, ya'ni 50% dan atigi ~2.2σ; 100+ konfiguratsiya ichida bu osongina tasodif.
- Oxirgi 6–12 oyni **hech qachon** ochmang; undan keyin 6–8 hafta real vaqtda paper/shadow trading.

**Model sozlamalari** (havo sharini pasaytirish): `num_leaves 8`, `max_depth 4`, `min_child_samples 300`, `colsample 0.7`, `reg_lambda 5`, early stopping (train'ning oxirgi 15% ichki validation). 500 ta daraxt / 31 leaf + early stopping'siz 42 ta shovqinli feature → overfit. Majburiy baseline: doimiy `0.5` va oddiy `LogisticRegression`.

---

## 7. Sayt arxitekturasi (Binance uchun)

Avval **tadqiqot/monitoring dashboard**, keyin (faqat eshiklardan o'tsa) savdo.

```
Binance REST + WebSocket
  ├─ klines 15m  ─┐
  ├─ fundingRate  │
  ├─ openInterest │──> PostgreSQL / TimescaleDB (UTC, idempotent upsert)
  ├─ longShortRatio
  └─ !forceOrders ┘
            │
    [ feature_service.py ]  ← BITTA kod, ham train ham inference uchun
            │               (training/serving skew'ning oldini oladi)
    ┌───────┴────────┐
    │                │
[offline trainer]  [FastAPI inference]
 walk-forward        har YOG'ILGAN shamda (o'rtasida emas)
 model registry      prob → kalibrlangan threshold → signal
 model.joblib        risk layer: max position, kunlik loss, kill-switch
 features.json       shadow mode = DEFAULT
 threshold + hash
 costs + data window
    │                │
    └───────┬────────┘
            ▼
   Web UI: sham grafigi + signallar + kalibratsiya egri chizig'i
           + feature drift (PSI/KS) + walk-forward hisoboti + paper PnL
            │
   [execution service] — alohida, IP whitelist, faqat order ruxsati,
                          withdrawal O'CHIQ, kalitlar hech qachon frontend'ga chiqmaydi
```

**Muhim qoidalar**

- Signal faqat sham **yopilgandan keyin** hisoblansin; kirish keyingi sham `open`'ida (siz buni to'g'ri qilgansiz — saqlang).
- Feature spetsifikatsiyasi, threshold, xarajat modeli va ma'lumot oralig'i model bilan birga versiyalansin (siz buni boshlagansiz — davom ettiring).
- Monitoring: feature PSI/KS drift, bashorat va realizatsiya kalibratsiyasi, jonli precision vs backtest, xarajatlar fakti. Drift → avtomatik `kill-switch`.
- Kalitlar: `.env` / secrets manager, read-only API default, order kalitlari alohida servisda, IP whitelist, notebook'ga parol yozmang.

**Go / No-Go eshiklari**

1. Walk-forward'da median net > 0 **va** musbat fold'lar ≥ 75% **va** `10 bps` xarajatda ham musbat.
2. Gross edge ≥ `2 × round-trip cost` (hozir 1.38 bps vs 10 bps — bu eshikdan o'tmaydi).
3. 6–8 hafta shadow/paper'da backtest bilan moslik.
4. Faqat juda kichik hajm, kunlik zarar limiti, keyin asta-sekin.

**Hozirgi holatda javob: NO-GO.** Saytni ajoyib analytics/dashboard sifatida qurish to'g'ri, lekin real order'larni ulash — yo'q.

---

## 8. Birinchi hafta rejasi

| Kun | Ish |
|---|---|
| 1 | Ma'lumotni Binance'dan qayta yuklab solishtirish; dublikat/gap/UTC/zero-volume tekshiruvi; DB parolini `.env`ga ko'chirish; `params`ni aniq yozish |
| 2 | `pipeline_v2.py`ni o'z Postgres'ingizga ulab ishga tushirish: `python pipeline_v2.py --table public.btcusdt_15m --cost-bps 10 --folds 10` |
| 3 | Faqat stasionar feature'lar bilan A/B/C o'rniga **bitta** to'plam; korrelyatsiya > 0.95 bo'lgan juftlarni o'chirish |
| 4 | Triple-barrier + meta-labeling; `y`ni xarajatdan keyingi foyda sifatida qayta aniqlash |
| 5 | 10 fold'li purged walk-forward; har fold natijasini JSON'ga yozish; Deflated Sharpe hisoblash |
| 6 | Tashqi ma'lumotlarni (funding, OI, long/short ratio, liquidations) yig'ishni qo'shish va ularning **qo'shimcha** qiymatini alohida ablatsiya bilan o'lchash |
| 7 | Natija musbat bo'lsa → shadow-mode servis + dashboard; bo'lmasa → saytni analytics vositasi sifatida qoldirib, volatillik/rejim bashoratiga o'tish |

---

## 9. Xulosa

Sizning kodingiz **yomon emas** — u ko'pchilik kripto ML loyihalaridan ancha intizomli. Muammo shundaki, u juda shovqinli savolga ("keyingi sham yashilmi?"), nostatsionar feature'lar bilan, xarajatlarni 5–10 barobar kam baholab javob izlayapti. Modelni emas, **savolni, feature'larni va xarajat modelini** almashtiring. Keyin "yutuqli model" o'rniga aniqroq maqsad qo'ying: *ko'p fold'da, real komissiyadan keyin, kichik lekin barqaror net edge + qat'iy risk boshqaruvi*. Bunga erishilmasa, eng to'g'ri qaror — savdo qilmaslik va saytni tahlil vositasi sifatida ishlatish.

---

### Fayllar

- `E:\OSH\OSH_QOLLANMA.md` — ushbu hujjat
- `E:\OSH\pipeline_v2.py` — stasionar feature + triple-barrier + purged walk-forward + real xarajat; `--synthetic` bilan o'zini tekshiradi
- `E:\OSH\pipeline_v2_report.json` — sintetik self-test hisoboti (namuna format)

```bash
cd E:\OSH
python pipeline_v2.py --synthetic --rows 80000 --folds 4 --cost-bps 10
python pipeline_v2.py --table public.btcusdt_15m --cost-bps 10 --folds 10 --out report.json
```

---

## Qo'shimcha (2026-11-13): tuzatmalar haqiqiy ma'lumotda ishga tushirildi

`pipeline_v2.py` 240 672 ta sham (`2020-01-01 → 2026-11-11 UTC`, 0 ta uzilish) bo'yicha,
eng so'nggi 10 ta purged walk-forward fold'da, 10 bps real xarajat bilan yuritildi.

| Ko'rsatkich | Eski notebook | Tuzatilgan pipeline |
|---|---|---|
| Median ROC AUC | 0.5139–0.5157 | **0.6524** |
| Brier / doimiy baseline | 0.2609 / ≈0.25 | 0.1434 / 0.1320 |
| Validatsiya | bitta 70/15/15 split | 10 ta xronologik fold |
| Jami savdo | — | 1 974 ta qog'oz savdo |
| Win rate @ 10 bps | 52–56% precision (xarajatsiz) | **40.63%** (95% CI 38.48–42.81%) |
| Median net / savdo | ≈ +1.38 bps gross | **−10.484 bps** |
| Musbat davrlar | — | **0 / 10** |
| Qaror | amalda NO-GO (xato xarajat bilan GO deb qabul qilingan) | **NO-GO** |

**Xulosa o'zgarmadi, lekin aniqlandi:** modelning tartiblash qobiliyati sezilarli yaxshilandi
(AUC 0.515 → 0.652), biroq bu asosan volatillik rejimini bashorat qilishdan keladi va
savdo foydasiga aylanmaydi. Hatto 0 bps xarajatda ham o'rtacha net `−0.35 bps`, ya'ni
yo'nalish bo'yicha edge yo'q. Bu "model yaxshilandi = foyda bor" degan xulosaning noto'g'ri
ekanini ko'rsatadi.

Natijalar va xavfsiz (paper-mode) dashboard:
**https://sherzoddd2002.github.io/osh-btcusdt-winrate-audit/** ·
repo: **https://github.com/Sherzoddd2002/osh-btcusdt-winrate-audit**
