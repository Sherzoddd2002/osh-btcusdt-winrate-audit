# OSH · BTCUSDT 15m — halol win-rate hisoboti

**Statik tadqiqot dashboard'i.** Tuzatilgan ML pipeline'ning haqiqiy ma'lumotlar bo'yicha
`purged walk-forward` natijalari. Hech qanday real buyurtma yuborilmagan, hech qanday kapital
xavf ostida emas.

> ⚠️ **Bu investitsiya maslahati emas.** Hech bir model foyda kafolatlamaydi. Barcha raqamlar
> tarixiy qog'oz savdo (backtest) bo'lib, kelajakdagi natijani kafolatlamaydi.
> `live_orders = false` — sayt faqat analytics uchun.

> **METODOLOGIK HOLAT — MUHIM:** ushbu branch'da v2/v3 tadqiqot kodi jiddiy qayta mustahkamlandi.
> Quyida ko'rsatilgan eski dashboard va `v3_results.json` raqamlari yangi kod bilan qayta
> hisoblanmagan; ular faqat tarixiy/arxiv natijalaridir. Yangi natijani faqat o'zingizning
> PostgreSQL ma'lumotingizda testlar o'tgach qayta yarating. Hozircha hech qanday strategiya
> GO deb tasdiqlanmaydi; jonli savdo o'chirilgan.

## Yangi hardening branch

- Branch: `research/v3-hardening-derivatives-data`
- Qattiq 15m interval / dublikat / OHLC / hajm validatsiyasi; qatorlarni jimgina tashlab yubormaydi.
- Train threshold endi test oynasidan top-5% rank qilish bilan almashtirilmagan: alohida train calibration tail ishlatiladi.
- Future OHLC barrier, bir shamda ikki to'siq tegsa konservativ stop, gap-through stop yomonroq open narxda.
- 2m cross-asset feature faqat source bar yopilib, BTC candle close vaqti kelganidan keyin mavjud deb olinadi.
- Audit import qilinganda DB'ga ulanmaydi; birinchi `NaT` diff uzilish deb sanalmaydi.
- Regression testlar va GitHub Actions CI qo'shildi.
- Eski news / ablation skripti non-causal bo'lgani uchun ataylab bloklandi.

---

## Natija (qisqa)

| Ko'rsatkich | Qiymat |
|---|---|
| Ma'lumot | 240 672 ta 15m sham, 2020-01-01 → 2026-11-11 UTC, 0 ta uzilish |
| Feature | 37 ta stasionar (log-return, rolling z-score, volatillik, sessiya) |
| Target | volatillikka asoslangan triple-barrier + xarajat chegarasi |
| Validatsiya | eng so'nggi **10 ta purged walk-forward fold**, har biri 90 kun |
| Jami qog'oz savdo | **1 974** |
| **Win rate @ 10 bps** | **40.63%** (Wilson 95% CI **38.48–42.81%**) |
| Yalpi win rate (0 bps) | 47.01% |
| Median net / savdo | **−10.484 bps** |
| Musbat davrlar | **0 / 10** (talab: ≥ 75%) |
| Median ROC AUC | 0.6524 |
| Brier | 0.1434 (doimiy baseline 0.1320 → kalibrlanmagan) |
| Benchmarkdan farq | −0.395 bps |
| **Qaror** | **NO-GO** — jonli savdo yo'q |

### Xarajat sezgirligi

| Round-trip xarajat | Win rate | O'rtacha net | Compounded |
|---|---|---|---|
| 0 bps (imkonsiz ideal) | 47.01% | −0.35 bps | −7.85% |
| 1 bps | 45.95% | −1.35 bps | −24.36% |
| 2 bps | 45.44% | −2.35 bps | −37.91% |
| 4 bps | 44.33% | −4.35 bps | −58.17% |
| 8 bps (futures taker) | 42.30% | −8.35 bps | −81.02% |
| **10 bps (asosiy)** | **40.63%** | **−10.35 bps** | **−87.21%** |
| 20 bps (spot taker) | 31.97% | −20.35 bps | −98.23% |

**Hatto 0 bps da ham o'rtacha net manfiy** — demak bu xarajat muammosi emas, yo'nalish
bo'yicha ishlatiladigan axborotda edge yo'q. Shuning uchun hech qanday komissiya darajasida
"yutuqli" bo'lmaydi.

---

## Nega AUC 0.65, lekin foyda yo'q?

Triple-barrier label yo'nalishni emas, **to'siq urilishi ehtimolini** bashorat qiladi. Model
bunda asosan volatillik rejimini o'rganadi (volatillik yuqori bo'lsa to'siqlar tezroq uriladi),
shu sababli AUC ko'tariladi. Bu statistik mahorat, lekin iqtisodiy edge emas:

- **AUC** = bashoratlar tartibi, savdo foydasini o'lchamaydi.
- **Brier** doimiy baseline'dan yomon → ehtimollar kalibrlanmagan.
- **Win rate + net bps** = haqiqiy savdo natijasi. Qaror shulardan chiqadi.
- Konservativ qoida: ikkala to'siq bitta sham ichida urilsa, **zarar** deb hisoblanadi
  (intrabar ma'lumot yo'q).

---

## Asl notebook'ga nisbatan nima o'zgardi

| Eski (`OSH MAIN.ipynb`) | Yangi (`pipeline_v2.py`) |
|---|---|
| Absolyut narx/volume feature'lari (nostatsionar) | Faqat log-return, nisbat, rolling z-score |
| Faqat bitta sham konteksti | 1/2/4/8/16/32/96 shamlik lag + volatillik + oqim |
| `month/dow/hour/mod_sin_cos` shovqini | Faqat 3 ta sessiya + weekend |
| "Keyingi sham yashilmi?" (UP 50.19%) | Triple-barrier: +1 / −1 / 0, xarajat chegarasi bilan |
| Bitta 70/15/15 split, holdout 4 marta ochilgan | 10 ta purged walk-forward fold, eng so'nggi davrlar |
| Threshold butun validatsiyadan keyin muzlatilgan | Threshold faqat har fold'ning **train** qismidan |
| 0–2 bps asosiy xarajat stsenariysi | **10 bps** real round-trip + 7 nuqtali sezgirlik |
| Accuracy / AUC / precision | Win rate + Wilson 95% CI + net bps + fold taqsimoti |
| `Brier 0.2609` (> 0.25) | `Brier 0.1434`, lekin hali ham baseline'dan yomon |

---

## Xavfsizlik kafolatlari ("tavakkal bo'lmagan")

- Sayt **to'liq statik**: bitta `index.html`, ichiga ma'lumotlar joylashtirilgan. Backend yo'q.
- **API kalit yo'q**, birja ulanishi yo'q, order yuborish kodi yo'q, hisob ma'lumotlari yo'q.
- Repozitoriyga faqat agregat natijalar joylandi — xom 240k qator, DB paroli yoki `.env` **yo'q**.
- Barcha savdolar tarixiy qog'oz savdo. Jonli rejim o'chirilgan.
- Sayt tashqi CDN, shrift yoki kuzatuv skripti yuklamaydi (oflayn ham ishlaydi).

---

## Qayta ishga tushirish (hardening branch)

> Quyidagi buyruqlar yangi branch kodiga tegishli. Eski reportlar avtomatik yangilanmaydi;
> ular faqat lokal qayta hisobdan keyin yangi natija bo'ladi.

### 1. Muhit va testlar

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m compileall -q .
python -m unittest discover -s tests -v
```

### 2. PostgreSQL audit

Parolni buyruq satriga yoki Git'ga yozmang. PostgreSQL `pgpass.conf` yoki maxfiy
environment variable'dan foydalaning.

```powershell
$env:OSH_DB_URL = "postgresql+psycopg2://postgres@127.0.0.1:5433/osh"
python audit_btc_integrity.py --table public.btcusdt_15m --out integrity_report.json
```

### 3. V3 kauzal tadqiqot

```powershell
python research_v3.py --table public.btcusdt_15m --cost-bps 10 --folds 5 --train-days 365 --test-days 90 --holdout-days 180 --out v3_results.json
```

Faqat BTC bazasini tekshirish:

```powershell
python research_v3.py --table public.btcusdt_15m --no-external --out v3_base_only.json
```

### 4. 1 kunlik selektiv signal sinovi (2% / 5% / 10%)

Savdo sonini ko'paytirishga emas, eng yuqori ishonchli signallarni ajratishga mo'ljallangan
alohida sinov. U **faqat 1d BASE** modelini tekshiradi va har bir walk-forward fold'da
modelni bir marta o'qitib, aynan o'sha modelning kalibrlash ehtimollaridan uchta threshold
oladi. Shu sababli 2%, 5% va 10% variantlari bir xil model va bir xil test davrlarida
taqqoslanadi.

PowerShell:

```powershell
$env:OSH_DB_URL = "postgresql+psycopg2://postgres@127.0.0.1:5433/osh"
.\.venv\Scripts\python.exe selectivity_sweep_1d.py --table public.btcusdt_15m --cost-bps 10 --selection-pcts 0.02,0.05,0.10 --folds 5 --train-days 365 --test-days 90 --holdout-days 180 --out selectivity_sweep_1d.json
```

Natijalar `selectivity_sweep_1d.json` fayliga yoziladi. Har bir variant uchun OOS va
alohida holdout win rate, Wilson 95% interval, net expectancy, Profit Factor, o'rtacha
net yutuq/zarar, payoff ratio, maksimal trade-sequence drawdown, ketma-ket zararlar,
fold natijalari va GO/NO-GO eshiklari ko'rsatiladi.

**Muhim metodologiya:** 2% / 5% / 10% — kalibrlash oynasidagi UP va DOWN ehtimollarining
har biri uchun alohida yuqori kvantil. Bu testda aynan shuncha savdo bo'ladi degani emas;
real signal va bajarilgan savdo ulushi alohida o'lchanadi. Threshold test va holdout
boshlanishidan oldin muzlatiladi. Bir nechta variantni ko'rib chiqqandan keyin holdout'da
eng yaxshisini tanlab olish mumkin emas — variantni development fold'laridan tanlab,
keyingi yangi forward davrida bir marta tasdiqlash kerak.

Bu script faqat tarixiy paper-research; real order yubormaydi. Hatto barcha GO
mezoni bajarilgan taqdirda ham, bu live trading uchun kafolat yoki ruxsat emas: keyingi
mustaqil forward sinov, ijro xarajatlari va risk nazorati alohida talab qilinadi.

`WIDE_GLD_AUX_2MIN.csv` ichidagi `datetime` 2-daqiqalik barning **open** vaqti
deb qabul qilinadi; kod qiymatni 2 daqiqadan keyin mavjud deb hisoblaydi. CSV timestamp'i
aslida close time bo'lsa, `--source-bar-minutes 0` ishlating. Semantika tekshirilmasdan
xasset natijasini dalil deb qabul qilmang.

### 5. $100 kapital: komissiya, slippage va funding stress-sinovi

Bu tajriba 1 kunlik BASE modelni turli ijro xarajatlari ostida qayta o'qitadi. U faqat
oldingi 10 bps natijasiga keyin komissiya qo'shmaydi: model barrier-label xarajati har
bir **komissiya + slippage** yig'indisi uchun qayta quriladi. Bir xil jami xarajatga ega
fee/slippage juftliklari bir xil fold modelidan foydalanadi.

Standart Regular User taxminlari (akkauntning shaxsiy Fee Rate jadvalini tekshiring):
maker **2 bps/side** (0.02%), taker **5 bps/side** (0.05%). Shuning uchun:
maker/maker = 4 bps round-trip; maker/taker = 7 bps; taker/taker = 10 bps.
Slippage 0/2/5 bps — bozor chuqurligidan o'lchangan fakt emas, balki foydalanuvchi
tanlaydigan round-trip stress farazlari. `--bnb-discount-pct 10` BNB chegirmasini
alohida tekshiradi; faqat USDⓈ-M Futures wallet'da komissiyani BNB bilan to'lashga
yetarli BNB bo'lsa qo'llanadi.

Funding `8h` deb taxmin qilinmaydi. Skript Binance public
`/fapi/v1/fundingRate` tarixidan `fundingTime`, `fundingRate`, `markPrice` qatorlarini
sahifalab yuklaydi va ularning to'liqligini tekshiradi. Funding long/short yo'nalishi,
pozitsiya muddati va settlement paytidagi mark narxiga qarab hisoblanadi. 15m sham
ichidagi aniq fill soniyasi ko'rinmagani uchun kirish/chiqish chegara shamidagi
funding to'lovi hisoblanadi, ehtimoliy funding krediti esa olinmaydi.

PowerShell:

```powershell
$env:OSH_DB_URL = "postgresql+psycopg2://postgres@127.0.0.1:5433/osh"
.\.venv\Scripts\python.exe capital_fee_stress_1d.py --table public.btcusdt_15m --fetch-funding --funding-csv .\binance_btcusdt_funding_history.csv --maker-fee-bps 2 --taker-fee-bps 5 --bnb-discount-pct 0 --slippage-bps 0,2,5 --selection-pcts 0.02,0.05,0.10 --position-caps 25,50,100 --initial-capital 100 --cash-reserve-pct 10 --folds 5 --train-days 365 --test-days 90 --holdout-days 180 --out capital_fee_stress_1d.json
```

Natija `capital_fee_stress_1d.json` ichida 3 fee rejimi × 3 slippage rejimi × 3
signal selektivligi bilan solishtiriladi. Har bir variant uchun OOS va untouched
holdout win rate/95% CI, Profit Factor, net expectancy, komissiya, slippage, funding
to'lovi va krediti, drawdown, hamda $25/$50/$100 pozitsiya-cap bilan $100 hisobning
yakuniy balansi beriladi.

`cash-reserve-pct 10` sabab $100 cap ssenariysi dastlab $90 nominal bilan boshlanadi;
$25 va $50 cap esa mos ravishda $25/$50 dan boshlanadi. Kapital tushsa pozitsiya
90% joriy balans bilan chegaralanadi; yuqoriga qarab tanlangan cap'dan oshmaydi.
Bu 1x, zaxirali qog'oz simulyatsiya, lekin exchange liquidation/maintenance margin,
order-book queue yoki maker order fill ehtimolini modellashtirmaydi. Maker/maker
natijasi maker fill kafolati emas.

Agar `fapi.binance.com` tarmoqdan ochilmasa, funding tarixini Binance'dan CSV qilib
olib `--funding-csv` bilan bering; funding ma'lumoti bo'lmasa skript uni nol deb
taxmin qilib GO bermaydi. Hisob ma'lumotlari va fee tier akkauntga bog'liq.

Rasmiy manbalar: [USDⓈ-M fee rate table](https://www.binance.com/en-BH/fee/futureFee),
[Binance Futures fees and funding](https://www.binance.com/en/support/faq/detail/98488a516eb84e3eb34605683dffd554),
[Funding rate history API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History).


### 6. V2 bazaviy pipeline

```powershell
python pipeline_v2.py --table public.btcusdt_15m --cost-bps 10 --folds 5 --train-days 365 --test-days 90 --out report.json
```

Sintetik self-test kod ishga tushishini tekshiradi, bozor edge'ini emas:

```powershell
python pipeline_v2.py --synthetic --rows 80000 --folds 4 --cost-bps 10
```

### 7. Eskirgan tajriba va eski natijalar

`ablation_experiment.py` non-causal test ranking va external timestamp muammolari sabab
ataylab o'chirilgan. `ablation_results.json`, `v3_results.json`, `index.html`,
`research-v3.html` va `external-audit.html` yangi hardening kodidan oldingi arxiv
natijalaridir. Qayta hisoblanmaguncha ularni yangi model natijasi sifatida ko'rsatmang.

To'liq nazariy qo'llanma: [`QOLLANMA.md`](QOLLANMA.md).

---

## Jonli sayt

- **Asosiy dashboard:** https://sherzoddd2002.github.io/osh-btcusdt-winrate-audit/
- **Tashqi ma'lumotlar auditi:** https://sherzoddd2002.github.io/osh-btcusdt-winrate-audit/external-audit.html
- **Research v3 (tuzatilgan):** https://sherzoddd2002.github.io/osh-btcusdt-winrate-audit/research-v3.html

GitHub Pages yoqilgan: source **`main` / `/ (root)`** (2026-11-13 dan; avvalgi
`gh-pages` manbasida legacy Jekyll build xatosi chiqqani uchun o'zgartirildi).
`main` va `gh-pages` branch'lari bir xil fayllarni saqlaydi — asosiy manba `main`,
shuning uchun o'zgartirishni kamida `main` ga yuboring.

`.nojekyll` fayli `_` bilan boshlanuvchi papkalarning o'tkazib yuborilishining oldini oladi.

---

## Tashqi ma'lumotlar auditi (2026-11-13)

["Ma'lumotlarimiz to'g'rimi? Oltin bilan bog'liqlik bormi? Yangiliklarni sonli feature qilib qo'shsak yordam beradimi?"](external-audit.html)
savollariga javob — o'lchov bilan, taxmin bilan emas:

| Savol | Javob |
|---|---|
| BTC shamlar tozami | **Ha**: 240 672 qator, 0 ta OHLC/VWAP/dublikat xatosi, 0 ta haqiqiy uzilish, 16 ta nol-hajmli sham. `open = oldingi close` 46.9% (sintetik emas), lag-1 avtokorr −0.00241 → real bozor. So'ngi close 73 980 USDT (2026-11-11 23:45 UTC). |
| Yangiliklar bazasi | **Yo'q, 100% sintetik**: `macro_data/news_raw.csv` — 2 703 qator, manba `SyntheticNewsAPI`, URL `synthetic-news.com`, 224 ta noyob sarlavha, 42% shablon. So'rov ham `gold OR federal reserve...`, BTC emas. |
| Oltin bilan bog'liqlik | **Bor, lekin faqat bir vaqtda**: GLD 15m korr +0.098, kunlik +0.128, SLV +0.131/+0.211, VIXY −0.336/−0.391. 30 kunlik rolling GLD oralig'i **−0.456…+0.583** — juda beqaror. |
| Oldindan aytadimi | **Yo'q**: 1h oldinga IC barcha 6 aktivda \|IC\| < 0.015 (mezon \|IC\| > 0.03). Eng yaxshisi VIXY +0.0126, IR 0.712, lekin IC eshigidan o'tmaydi. |
| Yangilik feature'lari | **Yo'q**: barcha IC \|<0.007\|, tasodifiy nazorat (`control_random` = −0.00124) bilan bir xil; 24h kechikishli varianti yana yomon. |
| Nazoratli 6-fold ablatsiya | Baza −11.24 bps → **+oltin/makro −8.26 bps (+2.98)**, **+yangiliklar −12.07 bps (−0.83)**, ikkalasi −11.19 bps. **Barcha konfiguratsiyalarda 0/6 musbat davr, NO-GO.** |
| Sizning eski makro sinovingiz | `edge_pnl_daily.csv` (1 925 kun): `ML_price` +0.0054%/kun (Sharpe 0.095) → `ML_all(+macro)` −0.0243%/kun (Sharpe −0.422); farq **t = −2.44**, ya'ni makro qo'shish sezilarli yomonlashtirgan. |

**Xulosa:** xasset feature'lari kichik, haqiqiy yaxshilanish berdi (+3 bps), lekin 10 bps
xarajatni yopish uchun yana +8 bps kerak; yangiliklar esa yomonlashtirdi. Birinchi navbatda
**kripto-native** ma'lumot (funding rate, open interest, taker long/short, likvidatsiyalar,
basis, order book) va **haqiqiy BTC yangiliklari** (UTC + `available_at`, dedup, manba og'irligi)
kerak. Yangiliklarni yo'nalish emas, **volatillik/rejim** bashorati uchun sinash oqilona.

### Qayta hisoblash

```bash
# xom ma'lumotlar E:\OSH da; boshqa joyda OSH_DATA_DIR ni o'rnating
set OSH_DATA_DIR=E:\OSH
python audit_btc_integrity.py       # -> integrity_report.json
python external_analysis.py         # -> external_analysis.json
python ablation_experiment.py       # -> ablation_results.json (pipeline_v2.py ni import qiladi)
python build_dashboard.py --report report.json --out .
```

`external-audit.html` ichiga natijalar JSON ko'rinishida joylashtirilgan (fetch yo'q, oflayn ishlaydi;
faqat grafiklar uchun ECharts CDN — yuklanmasa jadval tugmasi bilan barcha aniq raqamlar ochiladi).

**Manba va cheklovlar:** barcha hisob-kitoblar lokal OSH ma'lumotlari bo'yicha 2026-11-13 da
qilingan. Bu muhitdan `api.binance.com` / `fapi.binance.com` (502) va GDELT (SSL timeout) ga
ulanib bo'lmadi, shuning uchun **birja bilan to'g'ridan-to'g'ri solishtirish qilinmadi**; narx
darajasi faqat ikkilamchi ommaviy xulosalar bilan mosligi tekshirildi. Tashqi adabiyot/korrelyasiya
raqamlari ikkilamchi manba bo'lib, asl matni tekshirilmagan.

---

## Research v3 — to'liq tuzatilgan eksperiment (2026-11-13)

[`research-v3.html`](research-v3.html) — user'ning ilmiy rejasiga asoslangan to'liq tuzatilgan
tadqiqot tizimi. Barcha 5 ta oldingi xato tuzatildi, 3 ta gorizont × 2 ta konfiguratsiya × 5 fold
purged walk-forward, untouched 6 oylik holdout, block-bootstrap 95% CI, GO/NO-GO eshiklari.

### Natijalar (3 gorizont, 2 konfiguratsiya)

| Gorizont | Konfig | Median net (bps) | Musbat fold % | Profit Factor | Bootstrap 95% CI | Holdout net (bps) | Verdict |
|---|---|---|---|---|---|---|---|
| 1h | BASE | −8.76 | 0% | 0.61 | [−10.23, −7.01] | −9.50 | NO-GO |
| 1h | +XASSET | −8.19 | 0% | 0.61 | [−10.48, −7.00] | −8.73 | NO-GO |
| 4h | BASE | −11.56 | 20% | 0.71 | [−17.11, −5.65] | −11.12 | NO-GO |
| 4h | +XASSET | −13.13 | 0% | 0.68 | [−18.24, −8.22] | −8.54 | NO-GO |
| **1d** | **BASE** | **+7.62** | **60%** | **1.01** | **[−25.85, +15.15]** | **+4.96** | **NO-GO*** |
| **1d** | **+XASSET** | **+3.69** | **60%** | **1.04** | **[−7.91, +17.71]** | **−18.76** | **NO-GO*** |

`*` 1d gorizontda median net > 0 va Profit Factor > 1, lekin bootstrap CI pastki chegarasi
manfiy va musbat fold % 75% dan kam. Holdout'da +XASSET konfiguratsiyasi −18.76 bps — ya'ni
o'rganilgan edge holdout'da yetib bormadi. **Ehtiyot bo'ling: 222 savdo = juda oz, statistik
kuch yo'q.**

### Tuzatilgan 5 ta xato

1. Train threshold muzlatildi, test'da ishlatildi (endi to'g'ri)
2. Triple-barrier endi high/low bilan (faqat close emas)
3. Early stopping purged validation bilan (leakage yo'q)
4. Tashqi ma'lumotlar `available_at` + max yosh bilan (stale ffill yo'q)
5. 6 oylik untouched holdout (hech qachon tuning/qayta o'qitish uchun ochilmaydi)

### Lookback tekshiruvlari (barchasi ✓)

- `threshold_frozen_from_train_and_used`: ✓
- `causal_chronological_selection`: ✓
- `early_stopping_purged`: ✓
- `barriers_use_high_low`: ✓
- `external_staleness_capped`: ✓
- `untouched_holdout`: ✓

### Qayta ishga tushirish

```bash
python research_v3.py --table public.btcusdt_15m --cost-bps 10 --out v3_results.json
```

---

## Keyingi qadam (agar edge izlash davom etsa)

Faqat OHLCV bilan 15m yo'nalishda edge topish real emas. Qo'shish kerak:

- `fundingRate`, `openInterest` o'zgarishi, `takerlongshortRatio`, `topLongShortPositionRatio`
- `!forceOrders` (likvidatsiyalar), perpetual−spot basis, `ETH/BTC`, order book imbalance
- Boshqa maqsad: yo'nalish emas, **volatillik / rejim / execution** bashorati
- Meta-labeling: avval oddiy qoida nomzod chiqaradi, ML faqat "foydalimi?" deb javob beradi
- Ikki bosqichli arxitektura: Market Data → Regime Detection → Strategy Engine → ML Meta-Labeling → Risk & Execution
- GO eshiklari: median net > 0, musbat fold ≥ 75%, PF > 1, bootstrap 95% CI pastki > 0, benchmarkdan yaxshi, leakage = 0

Bularsiz eng to'g'ri qaror — savdo qilmaslik va saytni analytics vositasi sifatida ishlatish.

---

*OSH research · barcha raqamlar tarixiy paper-backtest · investitsiya maslahati emas.*
