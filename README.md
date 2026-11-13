# OSH · BTCUSDT 15m — halol win-rate hisoboti

**Statik tadqiqot dashboard'i.** Tuzatilgan ML pipeline'ning haqiqiy ma'lumotlar bo'yicha
`purged walk-forward` natijalari. Hech qanday real buyurtma yuborilmagan, hech qanday kapital
xavf ostida emas.

> ⚠️ **Bu investitsiya maslahati emas.** Hech bir model foyda kafolatlamaydi. Barcha raqamlar
> tarixiy qog'oz savdo (backtest) bo'lib, kelajakdagi natijani kafolatlamaydi.
> `live_orders = false` — sayt faqat analytics uchun.

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

## Qayta ishga tushirish

```bash
# 1) Haqiqiy PostgreSQL ma'lumoti bilan (parol pgpass.conf orqali o'qiladi)
python pipeline_v2.py --table public.btcusdt_15m --cost-bps 10 --folds 10 \
  --train-days 365 --test-days 90 --out report.json

# 2) Yoki hech qanday DB siz, sintetik self-test
python pipeline_v2.py --synthetic --rows 80000 --folds 4 --cost-bps 10

# 3) Dashboard'ni qayta yig'ish
python build_dashboard.py --report report.json --out .
```

`pipeline_v2.py` quyidagilarni qiladi:

1. `make_features()` — faqat `t` sham yopilishida ma'lum bo'lgan 37 ta stasionar feature
   (training/serving symmetry).
2. `triple_barrier_labels()` — `σ = std(log ret, 96) · √h` faqat **o'tmish** ma'lumotidan;
   to'siqlar `±σ`; `y=1` faqat maqsad xarajatdan oshganda; ustma-ust label'lar `1/h` og'irlikda.
3. `make_folds()` — xronologik expanding train + 6 sham embargo + 90 kunlik test,
   **eng so'nggi** 10 fold olinadi.
4. Threshold har fold'da faqat train oxiridan kalibrlanadi; testga tegilmaydi.
5. `non_overlapping()` — pozitsiyalar ustma-ust tushmaydi; barrier-exit PnL.
6. `wilson_ci()` — win rate uchun 95% ishonch oralig'i.
7. `GO` faqat: median net > 0 **va** musbat fold'lar ≥ 75% **va** median AUC > 0.52.

To'liq nazariy qo'llanma: [`QOLLANMA.md`](QOLLANMA.md).

---

## Jonli sayt

**https://sherzoddd2002.github.io/osh-btcusdt-winrate-audit/**

GitHub Pages allaqachon yoqilgan: source `gh-pages` branch / `/ (root)`, status `built`.
`main` va `gh-pages` branch'lari bir xil fayllarni saqlaydi; o'zgartirish kiritganda
ikkalasini ham yangilang (yoki Settings → Pages'da source'ni `main / root` ga o'tkazing).

`.nojekyll` fayli `_` bilan boshlanuvchi papkalarning o'tkazib yuborilishining oldini oladi.

---

## Keyingi qadam (agar edge izlash davom etsa)

Faqat OHLCV bilan 15m yo'nalishda edge topish real emas. Qo'shish kerak:

- `fundingRate`, `openInterest` o'zgarishi, `takerlongshortRatio`, `topLongShortPositionRatio`
- `!forceOrders` (likvidatsiyalar), perpetual−spot basis, `ETH/BTC`, order book imbalance
- Boshqa maqsad: yo'nalish emas, **volatillik / rejim / execution** bashorati
- Meta-labeling: avval oddiy qoida nomzod chiqaradi, ML faqat "foydalimi?" deb javob beradi

Bularsiz eng to'g'ri qaror — savdo qilmaslik va saytni analytics vositasi sifatida ishlatish.

---

*OSH research · barcha raqamlar tarixiy paper-backtest · investitsiya maslahati emas.*
