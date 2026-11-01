# Photo Shadow-Lift Depth/Shadow Prepress Engine (v1.4.4)

[Русский](#-описание-и-документация-russian) | [English](#-description-and-documentation-english)

---

## 🇷🇺 Shadow-Lift Depth/Shadow Prepress Engine

**Экспериментальный Python-инструмент (v1.4.4) для автоматизированной предпечатной подготовки сложных Low-Key изображений, фотографий и художественных работ (карты Таро, постеры, артбуки), предназначенных для цифровой и офсетной печати.**

Инструмент предназначен для подготовки к печати материалов с глубокими тенями. Основное внимание уделяется повышению видимости деталей геометрии в тенях через адаптивную тональную коррекцию в пространстве **Oklab**, пространственно-ориентированную обработку (MiDaS depth, structure tensor, guided filter) и **трёхрежимную симуляцию поведения принтера The Game Crafter** с опциональным мягким floor (`--soft-floor`) и **калибровкой кривой по физическому оттиску** (`calibrate.py`).

**Статус проекта:** экспериментальный инструмент предпечатной подготовки. Перед запуском тиража каждое обработанное изображение следует проверять индивидуально на физическом пробном отпечатке.

### 📐 Структура проекта

| Файл | Назначение | Зависимости |
|---|---|---|
| `shadow_lift_depth_shadow_prepress.py` | Основной движок препресса | torch, cv2, scipy, PIL, numpy |
| `calibrate.py` | Утилита калибровки TGC-фильтра по физическому оттиску | только PIL + numpy |
| `default_curve.json` | Параметры TGC-симулятора (опционально, авто-загружается) | — |
| `requirements.txt` | Зависимости Python | — |

### 🧠 Используемые модели и алгоритмы

Мы отказались от тяжёлых нейросетей для теней в пользу детерминированной математики, чтобы обеспечить 100% повторяемость и скорость для 78+ карт.

*   **Модель карты глубины:** **MiDaS_small** (Intel ISL, `midas_v21_small_256.pt`). Используется для извлечения 2.5D карты глубины (кэшируется в `_prepress_maps` с content-hash по SHA256 файла). Веса **верифицируются по SHA256** при каждом запуске (`70d6b9c8…d56fca`), при несовпадении — prompt / abort / `--accept-models`. Поддерживается локальная папка `./models/` рядом со скриптом.
*   **Модель карты теней:** **ClassicalShadowDetector** (Finlayson 2009 / Guo 2013). Детерминистичный алгоритм, анализирующий локальное падение освещённости.
*   **Sky Discriminator (v1.3.1):** Разделяет небо и глубокий тоннель по трём осям — глубина (`depth > 0.7`), абсолютная темнота (`lum < 0.10`) и **плоскость** (`texture_mask ≈ 0`). Тоннель имеет структуру (арки, кирпич), небо — нет. Это устраняет «замыливание» тоннелей и «выбеливание» неба.
*   **TGC-симулятор (v1.4.4):** 5-параметровая кривая (`black_clip`, `gamma`, `desat_thresh`, `desat_amount`, `scale_factor`), вынесенная в `default_curve.json`. Только `black_clip` калибруется по физическому оттиску. Остальные 4 — из обратного инжиниринга TGC web preview.

### 🔍 Проблема

При цифровой печати на картоне происходят **растискивание краски (dot gain)** и компрессия динамического диапазона. The Game Crafter (TGC) вычитает ~23 RGB (≈9% яркости) из каждой точки, дополнительно применяет гамму и десатурацию теней. Из-за этого Low-Key иллюстрации на физическом отпечатке теряют микроконтраст и схлопываются в сплошные чёрные пятна («подвалы»).

### 💡 Решение (v1.4.4)

Пайплайн реализует **интеллектуальную многослойную коррекцию**:

1.  **Oklab Color Space:** Работа в перцептивном пространстве. Яркость (`L`) поднимается без потери насыщенности (Colored Black).
2.  **Guided Filter & Structure Tensor:** Разделение на «Базу» и «Детали» без «мыла». Тензор структуры находит реальную геометрию (трубы, провода, складки), игнорируя цифровой шум.
3.  **Structural Preservation:** Сглаживание дефицита яркости в плоских зонах, защита цветных объектов.
4.  **TGC Compensation — 3 режима (`--tgc-lift MODE [STRENGTH]`):**
    *   **`off` (default):** Без компенсации.
    *   **`safe`:** Глобальный **hard floor** (`min_black + 16 × strength`, 30 RGB при strength=1.0). Гарантирует отсутствие чёрных клякс. Небо сереет — цена за безопасность.
    *   **`stress`:** **Per-pixel floor** с учётом depth и текстуры. Foreground получает полный floor, небо защищено (`floor = min_black`).
    *   **`STRENGTH`:** Множитель 0.0–2.0 (default 1.0).
5.  **Soft Floor (`--soft-floor RGB`, v1.4.1):** Мягкий (монотонный) floor, заменяющий жёсткий `np.maximum` в `safe`/`stress`. Сохраняет градиент в тенях ценой ≤ k RGB запаса безопасности. См. раздел «🎚 Что такое soft-floor».
6.  **Prepress Filter (`--prepress-filter`, v1.4.3):** Два файла для типографа — `*_prepress.png` (симуляция печати) и `*_prepress_risk.png` (полупрозрачный красный overlay на зонах, где фильтр съел видимую деталь). Модель **не откалибрована** по умолчанию, но может быть откалибрована (см. ниже).
7.  **Калибровка по физическому оттиску (`calibrate.py`, v1.4.4):** Отдельная утилита. Печатаете мишень (41 градация, референсные white/black патчи), сканируете, обрезаете до 900×1500 — `calibrate.py` подбирает `black_clip` для **вашего** принтера + бумаги. Результат: `my_scan.json` → `--custom-curve my_scan.json`.
8.  **Content-hash Cache:** Карты глубины инвалидируются при изменении содержимого файла (SHA256) или версии модели.

---

### 🖼 Результат работы / Result Showcase

Пример работы движка v1.4.4 (`--tgc-lift stress --debug`):

| Исходник (Input) | Output (без TGC) | Output (Safe, strength 1.0) | Output (Stress, strength 1.0) |
| :---: | :---: | :---: | :---: |
| <img src="docs/showcase_input.jpg" width="200"/> | <img src="docs/showcase_output.png" width="200"/> | <img src="docs/showcase_output_safe.png" width="200"/> | <img src="docs/showcase_output_stress.png" width="200"/> |

**Что видно:**
- **Output** — базовый Oklab-пайплайн: сочные тени, читаемая геометрия, без TGC-компенсации.
- **Safe** — глобальный hard floor. Небо становится серым, но CrushMask почти пустая. Гарантия печати ценой атмосферы.
- **Stress** — умный per-pixel floor. Небо тёмное, звёзды читаются, тоннель и foreground вытянуты.

> 💡 При `--soft-floor 2` картинка Safe остаётся почти такой же по безопасности, но тени перестают быть плоской «пластиковой» заливкой — появляется микроградиент в диапазоне 28–30 RGB.
> 💡 При `--prepress-filter` дополнительно сохраняются `*_prepress.png` (симуляция) и `*_prepress_risk.png` (красный overlay фильтр-индуцированных провалов).

---

### 🚀 Быстрый старт

#### Требования и зависимости

```bash
pip install -r requirements.txt
```

Или вручную:
```bash
pip install Pillow numpy scipy opencv-contrib-python torch torchvision
```

> **`calibrate.py` не требует torch/cv2** — только `Pillow` + `numpy`. Калибровку можно делать на любой машине.

### 🛠 Использование (CLI)

**1. Базовый препресс (без TGC-компенсации):**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_prepress/ -s 900x1500 --tgc-lift off
```

**2. Гарантия печати (Safe):**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_safe/ -s 900x1500 --tgc-lift safe
```

**3. Умная компенсация (Stress):**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_stress/ -s 900x1500 --tgc-lift stress
```

**4. Мягкая компенсация (Stress 0.7):**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_soft/ -s 900x1500 --tgc-lift stress 0.7
```

**5. Безопасный floor с сохранением градиента:**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_safe_soft/ -s 900x1500 --tgc-lift safe --soft-floor 2
```

**6. Prepress filter для типографа (симуляция + risk overlay):**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_safe/ -s 900x1500 \
    --tgc-lift stress --prepress-filter
```

**7. С калиброванной кривой:**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_safe/ -s 900x1500 \
    --tgc-lift stress --prepress-filter --custom-curve my_scan.json
```

---

### 🎯 Калибровка по физическому оттиску (`calibrate.py`)

**Зачем:** По умолчанию `black_clip = 0.09` (≈23 RGB) — это **обратный инжиниринг TGC web preview**, а не измерение на физическом принтере. Калибровка даёт **ваш** `black_clip` для **вашей** бумаги + принтера + сканера с точностью ±2–3 RGB.

**Как работает:** Мишень содержит 41 градацию (`0, 1, 2, …, 40`) + референсные white/black патчи. После печати и скана утилита нормализует каждую градацию относительно white/black патчей (убивает сканерный offset/gain) и находит **первую градацию выше noise floor**. Это и есть ваш `black_clip`.

#### Workflow (7 шагов)

```bash
# 1. Сгенерировать мишень (900×1500, 41 градация, референсные патчи)
python calibrate.py --generate-probe target.png

# 2. Напечатать target.png на целевой бумаге + целевой принтер
#    (тот же картон/бумага, что и для тиража)

# 3. Дать высохнуть (10+ минут для струйной печати)

# 4. Сканировать в PNG, ≥300 DPI, БЕЗ авто-коррекции в драйвере
#    (отключить «авто-уровни», «улучшение», «сделать ярче»)

# 5. Обрезать скан вручную до точных границ мишени (Photoshop/GIMP/Paint)
#    Привести к 900×1500, сохранить как PNG

# 6. Калибровка — fit по двум изображениям
python calibrate.py target.png my_scan.png
# → создаёт my_scan.json рядом со сканом
# → печатает лесенку 0..40, порог, confidence (high/medium/low)

# 7. Использовать в основном скрипте
python shadow_lift_depth_shadow_prepress.py -i cards/ -o out/ \
    --tgc-lift stress --prepress-filter --custom-curve my_scan.json
```

#### Что калибрует и что нет

| Параметр | Калибруется | Почему |
|---|---|---|
| `black_clip` | ✅ да | Порог — точка, сканер её не искажает |
| `gamma` | ❌ нет | Сканер искажает форму кривой |
| `desat_thresh` | ❌ нет | Требует цветной мишени |
| `desat_amount` | ❌ нет | Требует цветной мишени |
| `scale_factor` | ❌ нет | Сканерный gain смешан с printer response |

В JSON `curve_meta` каждый параметр помечен: `"calibrated": true/false` и `"source": "physical_scan" / "reverse_engineered"`. В логе основного скрипта при загрузке JSON видны все 5 значений с их статусом.

#### Дополнительные режимы `calibrate.py`

```bash
# Старые визуальные мишени (только для инспекции, не для калибровки)
python calibrate.py --generate-probe-legacy ./probes/
# → tgc_blackbox_probe_legacy.png + tgc_highlight_probe_legacy.png

# Записать default_curve.json (для авто-загрузки основным скриптом)
python calibrate.py --dump-default default_curve.json
```

#### Ограничения калибровки

- **Только для вашей бумаги.** Глянец ≠ картон. Для каждой бумаги — свой `my_scan.json`.
- **Сканер вносит вклад.** Относительные измерения компенсируют offset/gain, но ±2–3 RGB ошибки остаётся. Не колориметрия.
- **Не отменяет физический пробник.** Калибровка уточняет модель, но реальный оттиск всё равно нужен для валидации.

---

### 📋 Основные аргументы командной строки

| Аргумент | Флаг | По умолчанию | Описание |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | *Обязательный* | Путь к исходному файлу или папке с изображениями. |
| `--output` | `-o` | *Обязательный* | Путь для сохранения (папка или `.png` файл). |
| `--size` | `-s` | `None` | Целевой размер `ШИРИНАxВЫСОТА`. |
| `--min-black` | `-b` | `14` | Нижняя точка чёрного. |
| `--max-white` | `-w` | `245` | Верхняя точка белого. |
| `--tgc-lift` | | `off` | `MODE [STRENGTH]` — `off` / `safe` / `stress`. |
| `--soft-floor` | | `0.0` | Мягкий монотонный floor (0–20 RGB). Только с `safe`/`stress`. |
| `--custom-curve` | | `None` | **v1.4.4.** JSON от `calibrate.py`. Только для `stress`-симуляции. |
| `--prepress-filter` | | `False` | **v1.4.3.** `*_prepress.png` + `*_prepress_risk.png` + severity report. |
| `--lift-strength` | | `"auto"` | `auto` или `0.0–1.0`. |
| `--depth-protection` | | `"auto"` | `auto` или `0.0–1.0`. |
| `--detail-boost` | | `1.5` | Усиление HDR-геометрии. |
| `--prepress-hdr` | | `1.0` | Сжатие светов. |
| `--cmyk-compression` | | `0.0` | Защита теней от «грязного» CMYK. |
| `--debug` | | `False` | Отладочные карты для разработчиков. |
| `--no-shadow` | | `False` | Пропустить расчёт карты теней. |
| `--recompute-depth` | | `False` | Принудительный пересчёт карт глубины. |
| `--accept-models` | | `False` | **v1.4.2.** Продолжить при mismatch весов MiDaS. |
| `--log` | | `False` | Дублировать stderr в `shadow_lift_prepress.log`. |

> ⚠️ **`--generate-probe` удалён из основного скрипта** в v1.4.4. Используйте `python calibrate.py --generate-probe target.png` или `python calibrate.py --generate-probe-legacy ./probes/`.

---

### 🎚 Что такое `--soft-floor` и какую проблему он решает

#### Проблема: «пластилиновая» плоскость в тенях

Режим `safe` использует **hard clamp**: `L_new = np.maximum(L_new, floor_L)`. Каждый пиксель яркостью **ниже** floor (30 RGB при strength=1.0) превращается **ровно** в 30 RGB.

Если на изображении большие площади темнее floor — они все становятся одним сплошным пятном. Ноль градиента. Ноль объёма.

По реальному прогону на колоде из 23 карт Таро:

| Карта | Пикселей ниже floor |
|---|---|
| The Star | **55.5 %** |
| The Moon | **52.0 %** |
| The Lovers | **46.9 %** |
| Justice | **44.7 %** |
| The High Priestess | **43.2 %** |
| Wheel of Fortune | **42.8 %** |
| The Magician | **40.0 %** |

На этих картах **больше половины холста** в `safe` превращается в плоскую заливку ровно `#1E1E1E`.

#### Решение: монотонный мягкий floor

`--soft-floor k` заменяет `np.maximum` на гладкую функцию:

```
output = x + δ² / (δ + k),   где δ = max(floor − x, 0),   k = soft_floor
```

| Вход (x) | δ | Выход | Комментарий |
|---|---|---|---|
| 35 | 0 | 35.00 | выше floor — не тронут |
| 30 | 0 | 30.00 | ровно на floor — identity |
| 28 | 2 | 28.67 | мягкий лифт |
| 20 | 10 | 22.50 | сильный лифт |
| 0 | 30 | 27.19 | асимптота ≈ floor − k |

**Свойства:** монотонна (`dy/dx = k²/(δ+k)² > 0` везде), identity в точке floor, асимптота `floor − k`.

#### Как выбрать `k`

| `k` | Когда использовать |
|---|---|
| **0** (default) | Абсолютная гарантия. Hard clamp. |
| **2** | Рекомендуемый старт. Плоские пятна оживают. |
| **4** | Видимый градиент в глубоких тенях. Требует проверки CrushMask. |
| **6+** | Только если физический пробник подтвердил. |

> ⚠️ `--soft-floor` работает **только** с `--tgc-lift safe` или `--tgc-lift stress`. В `off` режиме флаг игнорируется.

---

### 📊 Диагностика

#### Для типографа: `--prepress-filter`

Сохраняет **два файла**:
- `*_prepress.png` — симуляция печати (stress-модель)
- `*_prepress_risk.png` — красный полупрозрачный overlay на пикселях, где **фильтр съел видимую деталь** (было `lum > 10`, стало `< 8`)

В логе:
```
[PREPRESS] Simulation saved: out/Card_prepress.png
[PREPRESS] Risk overlay saved: out/Card_prepress_risk.png
[PREPRESS] Filter-induced crush: 2,134 px (0.16%) — OK
[PREPRESS] ⚠️ Model NOT calibrated. Verify on physical proof before print.
```

Severity: `OK` (<1 %), `MEDIUM` (1–5 %), `HIGH` (>5 %).

**Красное = фильтр виноват.** Пиксели, которые были тёмными **до** фильтра (`lum ≤ 10`), не красятся — это художественный замысел или вопрос к `--tgc-lift`.

#### Для разработчиков: `--debug`

Сохраняет 8 технических файлов:
- `*_dbg_oklab_L_original.png`, `*_dbg_oklab_L_lifted.png`
- `*_dbg_target_map.png`, `*_dbg_classical_shadow.png`
- `*_dbg_protection_mask.png`, `*_dbg_texture_mask.png`
- `*_fake_TCG_{SAFE|STRESS}.png`, `*_dbg_TCG_CrushMask_{MODE}.png`

Плюс при `--log` — `[FLOOR-DIAG]`:
```
[FLOOR-DIAG] SAFE: 702,186 px (52.01%) below floor (L < 0.2350) BEFORE clamp
[FLOOR-DIAG]   OKLAB range: [-0.2737, 0.2340] | mean=0.1728 std=0.0322
[FLOOR-DIAG]   ≈RGB range: [-68..30] → collapse to 30
[FLOOR-DIAG]   max deficit=0.5088 OKLAB (≈98 RGB) | severe (deficit>0.05): 34.61%
[FLOOR-DIAG]   ⚠️ HIGH variation — soft floor would help
```

- `below floor` > 25 % и `severe` > 10 % → кандидат на `--soft-floor 2`
- `⚠️ HIGH variation` → soft-floor даст заметный градиент

С soft-floor в логе:
```
   soft-floor:           2.0 RGB (monotonic, min≈28 RGB)
  [TGC-SAFE] soft floor applied: k=2.0 RGB (monotonic, min≈28 RGB)
```

С `--custom-curve`:
```
  [CURVE] Loaded: my_scan.json (profile=my_scan)
  [CURVE]   black_clip     = 0.0510  (calibrated)
  [CURVE]   gamma          = 1.2500  (from file)
  [CURVE]   desat_thresh   = 0.2500  (from file)
  [CURVE]   desat_amount   = 0.8000  (from file)
  [CURVE]   scale_factor   = 0.9500  (from file)
  [CURVE] ⚠️ Only black_clip calibrated. Other params from reverse engineering.
```

---

### ⚠️ Известные ограничения

- **Гладкий тоннель** (без текстуры): `texture_mask ≈ 0` → может быть классифицирован как небо → не лифтнётся в `stress`. Обход: `--tgc-lift safe` или `--tgc-lift stress 1.5`.
- **Плотная облачность**: `texture_mask > 0.5` → небо может быть классифицировано как foreground → выбелится. Обход: `--tgc-lift off`.
- **MiDaS в замкнутых пространствах** может давать высокую depth у стен — `foreground_gate` частично компенсируется `flat_sky_gate`.
- **`--soft-floor` не даёт 100% гарантии**: минимум ≈ `floor − k`. Для абсолютной гарантии — `--soft-floor 0`.
- **TGC-симулятор консервативен по умолчанию**: `black_clip = 0.09` (≈23 RGB) — обратный инжиниринг web preview, **не** измерение. Калибруется через `calibrate.py` (v1.4.4). До калибровки значения `SAFE_STRESS_OFFSET` и `--soft-floor` стоит проверять на физическом пробнике.
- **Калибровка только по `black_clip`**: `gamma`, `desat`, `scale` остаются из обратного инжиниринга. Сканер не позволяет разделить их вклад.
- **Калибровка привязана к бумаге**: глянец ≠ картон. Для каждой бумаги — свой `my_scan.json`.

---

## 📜 Лицензия / License

Проект распространяется под лицензией **MIT License**. Подробности в [LICENSE](LICENSE).

---

## ⚠️ Дисклеймер

**Этот проект представляет собой экспериментальный инструментарий. Он не является сертифицированным решением для подготовки файлов к печати.**

Результат зависит от исходного изображения. Экранный предпросмотр и симулятор фильтра TGC не дают окончательной гарантии. **Обязательно проверяйте результат на физическом пробном отпечатке до запуска тиража.**

**Используйте инструмент на своё усмотрение.**

---

## 🇬🇧 Shadow-Lift Depth/Shadow Prepress Engine

**An experimental Python toolkit (v1.4.4) for automated prepress processing of complex low-key images, photographs, and artwork intended for digital and offset printing.**

The toolkit focuses on improving shadow detail visibility through adaptive tonal correction in the Oklab color space, spatially guided processing (MiDaS depth, structure tensor, guided filter), and **three-mode simulation of The Game Crafter printer behavior** with optional soft-floor (`--soft-floor`) and **physical-print calibration** (`calibrate.py`).

**Project status:** Experimental prepress tool. Processed images should be reviewed individually on physical proofs before final production.

### 📐 Project Structure

| File | Purpose | Dependencies |
|---|---|---|
| `shadow_lift_depth_shadow_prepress.py` | Main prepress engine | torch, cv2, scipy, PIL, numpy |
| `calibrate.py` | TGC simulator curve calibration from physical print | PIL + numpy only |
| `default_curve.json` | TGC simulator parameters (optional, auto-loaded) | — |
| `requirements.txt` | Python dependencies | — |

### 🧠 Models and Algorithms Used

We abandoned heavy neural networks for shadows in favor of deterministic mathematics to ensure 100% reproducibility and speed for 78+ cards.

*   **Depth Map Model:** **MiDaS_small** (Intel ISL, `midas_v21_small_256.pt`). Extracts a 2.5D depth map (cached in `_prepress_maps` with SHA256 content-hash). Weights are **verified by SHA256** on every launch (`70d6b9c8…d56fca`); on mismatch: prompt / abort / `--accept-models`. Local `./models/` folder next to the script is supported.
*   **Shadow Map Model:** **ClassicalShadowDetector** (Finlayson 2009 / Guo 2013). Deterministic algorithm analyzing local illumination drop.
*   **Sky Discriminator (v1.3.1):** Separates sky from deep tunnels along three axes — depth (`depth > 0.7`), absolute darkness (`lum < 0.10`), and **flatness** (`texture_mask ≈ 0`).
*   **TGC Simulator (v1.4.4):** 5-parameter curve (`black_clip`, `gamma`, `desat_thresh`, `desat_amount`, `scale_factor`) externalized to `default_curve.json`. Only `black_clip` is calibratable from physical prints. The other 4 remain from TGC web preview reverse engineering.

### 🔍 The Problem

During digital printing on cardstock, physical phenomena like **dot gain** and dynamic range compression occur. TGC subtracts ~23 RGB (~9% luminance), then applies gamma and shadow desaturation. Low-key illustrations lose micro-contrast and collapse into flat black patches ("crushed blacks").

### 💡 The Solution (v1.4.4)

1.  **Oklab Color Space:** Perceptual brightness lifting without saturation loss.
2.  **Guided Filter & Structure Tensor:** "Base" + "Detail" split without blurring.
3.  **Structural Preservation:** Prevents dark walls from becoming blurry blobs.
4.  **TGC Compensation — 3 modes (`--tgc-lift MODE [STRENGTH]`):**
    *   `off` (default): no compensation.
    *   `safe`: global hard floor = `min_black + 16 × strength` (30 RGB @ 1.0).
    *   `stress`: per-pixel floor — foreground full floor, sky protected.
5.  **Soft Floor (`--soft-floor RGB`, v1.4.1):** Monotonic soft-floor preserving gradient at the cost of ≤ k RGB safety margin.
6.  **Prepress Filter (`--prepress-filter`, v1.4.3):** Two typographer-facing files — `*_prepress.png` (simulation) and `*_prepress_risk.png` (semi-transparent red overlay on filter-induced crush).
7.  **Physical-Print Calibration (`calibrate.py`, v1.4.4):** Print a 41-step target, scan it, crop to 900×1500 — `calibrate.py` fits `black_clip` for **your** printer + paper. Output: `my_scan.json` → `--custom-curve my_scan.json`.
8.  **Content-hash Cache:** Depth maps invalidate on content or model version change.

---

### 🖼 Result Showcase

Example pipeline run (v1.4.4, `--tgc-lift stress --debug`):

| Input | Output (no TGC) | Output (Safe, 1.0) | Output (Stress, 1.0) |
| :---: | :---: | :---: | :---: |
| <img src="docs/showcase_input.jpg" width="200"/> | <img src="docs/showcase_output.png" width="200"/> | <img src="docs/showcase_output_safe.png" width="200"/> | <img src="docs/showcase_output_stress.png" width="200"/> |

> 💡 With `--soft-floor 2` the Safe output keeps its safety but shadows stop looking plastic.
> 💡 With `--prepress-filter` you additionally get `*_prepress.png` (simulation) and `*_prepress_risk.png` (red overlay on filter-induced crush).

---

### 🚀 Quick Start

#### Requirements

```bash
pip install -r requirements.txt
```

> **`calibrate.py` does not require torch/cv2** — only `Pillow` + `numpy`. Calibration can be done on any machine.

### 🛠 Usage (CLI)

**1. Base prepress (no TGC):**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_prepress/ -s 900x1500 --tgc-lift off
```

**2. Print guarantee (Safe):**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_safe/ -s 900x1500 --tgc-lift safe
```

**3. Smart compensation (Stress):**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_stress/ -s 900x1500 --tgc-lift stress
```

**4. Soft compensation (Stress 0.7):**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_soft/ -s 900x1500 --tgc-lift stress 0.7
```

**5. Safe floor with preserved gradient:**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_safe_soft/ -s 900x1500 --tgc-lift safe --soft-floor 2
```

**6. Prepress filter for the print shop:**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_safe/ -s 900x1500 \
    --tgc-lift stress --prepress-filter
```

**7. With calibrated curve:**
```bash
python shadow_lift_depth_shadow_prepress.py -i ./cards_raw/ -o ./cards_safe/ -s 900x1500 \
    --tgc-lift stress --prepress-filter --custom-curve my_scan.json
```

---

### 🎯 Physical-Print Calibration (`calibrate.py`)

**Why:** The default `black_clip = 0.09` (≈23 RGB) is a **reverse-engineering of TGC web preview**, not a measurement of the physical printer. Calibration gives **your** `black_clip` for **your** paper + printer + scanner with ±2–3 RGB accuracy.

**How:** The target has 41 grayscale steps (`0, 1, 2, …, 40`) plus reference white/black patches. After print + scan, the tool normalizes each step relative to the ref patches (kills scanner offset/gain) and finds the **first step above the noise floor** — that's your `black_clip`.

#### Workflow (7 steps)

```bash
# 1. Generate target
python calibrate.py --generate-probe target.png

# 2. Print on target paper + target printer (same as the production batch)

# 3. Let it dry (10+ minutes for inkjet)

# 4. Scan to PNG, ≥300 DPI, NO auto-correction in the scanner driver

# 5. Manually crop the scan to exact target bounds, resize to 900×1500, save as PNG

# 6. Fit curve from the two images
python calibrate.py target.png my_scan.png
# → writes my_scan.json next to the scan
# → prints the 0..40 ladder, threshold, confidence (high/medium/low)

# 7. Use in the main script
python shadow_lift_depth_shadow_prepress.py -i cards/ -o out/ \
    --tgc-lift stress --prepress-filter --custom-curve my_scan.json
```

#### What is calibrated and what isn't

| Parameter | Calibrated | Why |
|---|---|---|
| `black_clip` | ✅ yes | Threshold is a point, scanner doesn't warp it |
| `gamma` | ❌ no | Scanner warps curve shape |
| `desat_thresh` | ❌ no | Needs a color target |
| `desat_amount` | ❌ no | Needs a color target |
| `scale_factor` | ❌ no | Scanner gain mixed with printer response |

In JSON, `curve_meta` tags each parameter with `"calibrated": true/false` and `"source": "physical_scan" / "reverse_engineered"`. The main script prints all 5 values with their status on load.

#### Extra `calibrate.py` modes

```bash
# Legacy visual probes (inspection only, not used by calibration)
python calibrate.py --generate-probe-legacy ./probes/
# → tgc_blackbox_probe_legacy.png + tgc_highlight_probe_legacy.png

# Dump default_curve.json (auto-loaded by main script if present)
python calibrate.py --dump-default default_curve.json
```

#### Calibration limits

- **Paper-specific.** Gloss ≠ cardstock. Each paper gets its own `my_scan.json`.
- **Scanner contributes.** Relative normalization cancels offset/gain, but ±2–3 RGB remains. Not colorimetry.
- **Does not replace a physical proof.** Calibration refines the model; physical prints remain the ground truth.

---

### 📋 CLI Command Parameters

| Parameter | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | *Required* | Path to input file or folder. |
| `--output` | `-o` | *Required* | Path for output (folder or specific `.png`). |
| `--size` | `-s` | `None` | Target size `WIDTHxHEIGHT`. |
| `--min-black` | `-b` | `14` | Minimum black point. |
| `--max-white` | `-w` | `245` | Maximum white point. |
| `--tgc-lift` | | `off` | `MODE [STRENGTH]` — `off` / `safe` / `stress`. |
| `--soft-floor` | | `0.0` | Monotonic soft-floor (0–20 RGB). Only with `safe`/`stress`. |
| `--custom-curve` | | `None` | **v1.4.4.** JSON from `calibrate.py`. Applies to `stress` simulation. |
| `--prepress-filter` | | `False` | **v1.4.3.** `*_prepress.png` + `*_prepress_risk.png` + severity report. |
| `--lift-strength` | | `"auto"` | `auto` or `0.0–1.0`. |
| `--depth-protection` | | `"auto"` | `auto` or `0.0–1.0`. |
| `--detail-boost` | | `1.5` | HDR geometry boost. |
| `--prepress-hdr` | | `1.0` | Highlight compression. |
| `--cmyk-compression` | | `0.0` | CMYK gamut compression in shadows. |
| `--debug` | | `False` | Developer debug maps. |
| `--no-shadow` | | `False` | Skip shadow map. |
| `--recompute-depth` | | `False` | Force recompute depth maps. |
| `--accept-models` | | `False` | **v1.4.2.** Proceed on MiDaS weight SHA256 mismatch. |
| `--log` | | `False` | Mirror stderr to `shadow_lift_prepress.log`. |

> ⚠️ **`--generate-probe` was removed from the main script in v1.4.4.** Use `python calibrate.py --generate-probe target.png` or `python calibrate.py --generate-probe-legacy ./probes/`.

---

### 🎚 What is `--soft-floor` and what problem does it solve

#### The problem: "plasticine" flatness in shadows

`safe` mode uses a **hard clamp**: `L_new = np.maximum(L_new, floor_L)`. Every pixel below the floor (30 RGB @ 1.0) becomes **exactly** 30 RGB.

Real run on a 23-card Tarot deck (`[FLOOR-DIAG]`):

| Card | Pixels below floor |
|---|---|
| The Star | **55.5 %** |
| The Moon | **52.0 %** |
| The Lovers | **46.9 %** |
| Justice | **44.7 %** |
| The High Priestess | **43.2 %** |

More than half the canvas becomes flat `#1E1E1E`.

#### The solution: monotonic soft-floor

```
output = x + δ² / (δ + k),   where δ = max(floor − x, 0),   k = soft_floor
```

| Input (x) | δ | Output | Comment |
|---|---|---|---|
| 35 | 0 | 35.00 | above floor — untouched |
| 30 | 0 | 30.00 | exactly on floor — identity |
| 28 | 2 | 28.67 | gentle lift |
| 20 | 10 | 22.50 | strong but not full |
| 0 | 30 | 27.19 | asymptote ≈ floor − k |

**Properties:** monotonic (`dy/dx = k²/(δ+k)² > 0` everywhere), identity at floor, asymptotic to `floor − k`.

#### Choosing `k`

| `k` | When to use |
|---|---|
| **0** (default) | Absolute guarantee. Hard clamp. |
| **2** | Recommended start. Flat patches come alive. |
| **4** | Visible gradient in deep shadows. Check CrushMask. |
| **6+** | Only if a physical proof confirms. |

> ⚠️ `--soft-floor` works **only** with `--tgc-lift safe` or `stress`. In `off` mode the flag is ignored.

---

### 📊 Diagnostics

#### For the print shop: `--prepress-filter`

Produces **two files**:
- `*_prepress.png` — print simulation (stress model)
- `*_prepress_risk.png` — semi-transparent red overlay on pixels where the **filter ate a visible detail** (was `lum > 10`, became `< 8`)

Log:
```
[PREPRESS] Filter-induced crush: 2,134 px (0.16%) — OK
[PREPRESS] ⚠️ Model NOT calibrated. Verify on physical proof before print.
```

Severity: `OK` (<1 %), `MEDIUM` (1–5 %), `HIGH` (>5 %).

**Red = the filter is the culprit.** Pixels that were already dark **before** filtering (`lum ≤ 10`) are not painted — they are either artistic intent or a `--tgc-lift` concern.

#### For developers: `--debug`

8 technical files:
- `*_dbg_oklab_L_original.png`, `*_dbg_oklab_L_lifted.png`
- `*_dbg_target_map.png`, `*_dbg_classical_shadow.png`
- `*_dbg_protection_mask.png`, `*_dbg_texture_mask.png`
- `*_fake_TCG_{SAFE|STRESS}.png`, `*_dbg_TCG_CrushMask_{MODE}.png`

Plus `[FLOOR-DIAG]` when `--log` is used. With `--custom-curve`:
```
  [CURVE] Loaded: my_scan.json (profile=my_scan)
  [CURVE]   black_clip     = 0.0510  (calibrated)
  [CURVE]   gamma          = 1.2500  (from file)
  [CURVE] ⚠️ Only black_clip calibrated. Other params from reverse engineering.
```

---

### ⚠️ Known Limitations

- **Smooth tunnel** (no visible texture): `texture_mask ≈ 0` → may be classified as sky. Workaround: `--tgc-lift safe`.
- **Heavy overcast**: `texture_mask > 0.5` → sky may be classified as foreground. Workaround: `--tgc-lift off`.
- **MiDaS in enclosed spaces** may give high depth to walls.
- **`--soft-floor` is not a 100% guarantee**: minimum ≈ `floor − k`. Use `--soft-floor 0` for absolute guarantee.
- **TGC simulator defaults are conservative**: `black_clip = 0.09` is reverse-engineered from the web preview, not a measurement. Calibrate via `calibrate.py` (v1.4.4) for your printer + paper.
- **Only `black_clip` is calibrated**: `gamma`, `desat`, `scale` remain from reverse engineering. Scanner cannot separate them from printer response.
- **Calibration is paper-specific**: gloss ≠ cardstock. Each paper gets its own `my_scan.json`.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE).

---

## ⚠️ Disclaimer

**This project is an experimental image-preprocessing toolkit. It is not a certified print-preparation solution.**

Results may vary. On-screen previews and the TGC filter simulator cannot fully predict a physical print. **Always validate final settings with a physical proof before producing a full batch.**

**Use at your own discretion.**