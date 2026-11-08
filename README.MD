Я проверил историю. В самом первом `README.MD` (который был в начале нашего диалога) был блок «Result Showcase», но там было **две таблицы**: одна на 3 изображения (Input | Depth | Output), другая на 2 (Input | Output). 

Однако, судя по актуальному коду (`create_qa_sheet`), движок генерирует **4-панельный QA-лист**: `Input | Depth | Shadow | Output`. Именно эта таблица на 4 изображения лучше всего демонстрирует работу пайплайна v1.7.4 с флагом `--smart-preserve`.

Вот полный, актуальный `README.md`, в который интегрирована таблица с четырьмя изображениями (в русской и английской версиях). Вы можете полностью скопировать его в свой репозиторий.

```markdown
# Photo Shadow-Lift Depth/Shadow Prepress Engine (v1.7.4)

[Русский](#-описание-и-документация-russian) | [English](#-description-and-documentation-english)

---

## 🇷🇺 Shadow-Lift Depth/Shadow Prepress Engine

**Экспериментальный Python-инструмент (v1.7.4 — Intelligent Arcana & Smart-Preserve Edition) для автоматизированной предпечатной подготовки сложных Low-Key изображений, фотографий и художественных работ (карты Таро, постеры, артбуки), предназначенных для цифровой и офсетной печати.**

Инструмент предназначен для подготовки к печати материалов с глубокими тенями. Основное внимание уделяется повышению видимости деталей геометрии в тенях через адаптивную тональную коррекцию в пространстве **Oklab**, пространственно-ориентированную обработку (MiDaS depth, structure tensor, guided filter), семантический анализ темных сцен (**Dark Scene Analyzer**) и **трёхрежимную симуляцию поведения принтера The Game Crafter (TGC)**.

**Статус проекта:** экспериментальный инструмент предпечатной подготовки. Перед запуском тиража каждое обработанное изображение следует проверять индивидуально на физическом пробном отпечатке.

---

### 📐 Структура проекта

| Файл / Папка | Назначение | Зависимости |
|---|---|---|
| `shadow_lift_prepress.py` | Основной движок препресса v1.7.4 | torch, cv2, scipy, PIL, numpy |
| `dark_scene/` | Семантический модуль детекции темных сцен (`--smart-preserve`) | cv2, scipy, numpy |
| `dark_scene/darkness_intent.py` | Анализатор намерений региона (Sky Prior & Spatial Awareness) | numpy |
| `dark_scene/mask_generator.py` | Генератор мягких непрерывных масок защищаемых зон | cv2, scipy, numpy |
| `check_models.py` | Инспектор моделей, проверка SHA256 и загрузчик весов MiDaS | только стандартная библиотека |
| `calibrate.py` | Утилита калибровки TGC-фильтра по физическому оттиску | только PIL + numpy |
| `default_curve.json` | Параметры TGC-симулятора (авто-загружается) | — |
| `requirements.txt` | Зависимости Python | — |

---

### 🖼 Результат работы / Result Showcase

Пример 4-панельного QA-листа, который генерируется движком с флагом `--qa-dir`. Слева направо: исходное изображение, карта глубины MiDaS, карта теней и финальный результат препресса.

| Исходное изображение (Input) | Карта глубины (Depth) | Карта теней (Shadow) | Результат препресса (Output) |
| :---: | :---: | :---: | :---: |
| <img src="docs/showcase_input.jpg" width="200"/> | <img src="docs/showcase_depth.png" width="200"/> | <img src="docs/showcase_shadow.png" width="200"/> | <img src="docs/showcase_output.jpg" width="200"/> |

*(Примечание: замените пути `docs/...` на реальные имена ваших файлов в репозитории).*

---

### 🧠 Ключевые алгоритмы и обновления v1.7.4

* **Dark Scene Analyzer v2.1 (`--smart-preserve`):** Интеллектуальный анализатор намерений темных сцен с поддержкой **Sky Prior** (пространственная фильтрация верхнего полушария кадра). Защищает глубокий космос, вакуум и градиенты ночного неба от искусственного «просветления» и полошения.
  * **Patch 1 (Ground Rescue):** Отвязка `target_map` от `depth_protection` в зонах `RECOVER` (передний план и детали земли восстанавливаются в полном объеме).
  * **Patch 2 (Smoke Killer):** Жесткий пол TGC и RGB-смещения мягко нейтрализуются в зонах `PRESERVE`, устраняя эффект «серого дыма» и бандинг на физическом оттиске.
* **Oklab Continuous Brightness Gate (Идея Б):** Лифтинг и компенсация пола TGC управляются непрерывным пространственным шлюзом яркости по L-каналу Oklab с гауссовым сглаживанием ($\sigma=15.0$). Это предотвращает возникновение ступенчатых градиентов («небесных швов») между небом и боковыми объектами.
* **Модель карты глубины:** **MiDaS_small** (Intel ISL, `midas_v21_small_256.pt`). Используется для извлечения 2.5D карты глубины (кэшируется в `_prepress_maps` с SHA256 content-hash). Веса **верифицируются по SHA256** при каждом запуске.
* **Continuous Auto-Tuner (`--auto`):** Автономный классификатор архетипов сцен (включая специализированные профили для оккультных металлов, огня/факелов в полутьме, виньетирования и ночных небес).
* **Blue-Noise Dithering & Debanding (`--deband-strength`):** Внедрение высокочастотного синего шума в сочетании с динамической маской плато по локальной дисперсии для борьбы с бандингом на гладких градиентах.
* **Безопасная обработка абсолютного черного:** Защита от деления на ноль при расчете `lift_ratio`.

---

### 🔍 Проблема

При цифровой печати на картоне происходят **растискивание краски (dot gain)** и компрессия динамического диапазона. The Game Crafter (TGC) вычитает ~23 RGB (≈9% яркости) из каждой точки, дополнительно применяет гамму и десатурацию теней. Из-за этого Low-Key иллюстрации на физическом отпечатке теряют микроконтраст и схлопываются в сплошные чёрные пятна («подвалы»).

---

### 💡 Решение (Пайплайн v1.7.4)

1. **Oklab Color Space & Blue-Noise Dithering:** Работа в перцептивном пространстве Oklab с сине-шумовым дизерингом для предотвращения бандинга.
2. **Smart Preservation (`--smart-preserve`):** Семантический шлюз, предотвращающий подъём фона в глубоких зонах ночной тьмы без появления виниловых стыков и рваных краев.
3. **TGC Compensation — 3 режима (`--tgc-lift MODE [STRENGTH]`):**
   * **`off` (default):** Без компенсации.
   * **`safe`:** Глобальный **hard floor** (`min_black + 16 × strength`).
   * **`stress`:** **Per-pixel floor** с учётом depth, текстуры и семантики Dark Scene. Foreground получает полный floor, ночное небо защищено.
4. **Soft Floor (`--soft-floor RGB`):** Мягкий (монотонный) floor, заменяющий жёсткий `np.maximum`. Сохраняет плавность тоновых переходов.
5. **Black Anchor (`--black-anchor FLOAT`):** Точечная защита истинно черных пикселей (Dmax) от нежелательного подъема.

---

### 🖼 Результат работы / Result Showcase

Пример работы движка v1.5.2 в различных режимах (сгенерировано через `generate_showcase.bat`):

| Режим и Параметры | Описание | Результат |
| :--- | :--- | :--- |
| **Auto (No TGC)**<br>`--auto --tgc-lift off` | Чистый sRGB. Вытаскивает детали, но не компенсирует ограничения печати. | <img src="docs/showcase/01_auto_tgc_off.png" width="200"/> |
| **TGC Safe**<br>`--auto --tgc-lift safe 1.0` | Глобальный жесткий пол. Гарантирует отсутствие черных клякс, но может "убить" атмосферу. | <img src="docs/showcase/02_auto_tgc_safe.png" width="200"/> |
| **TGC Stress (Recommended)**<br>`--auto --tgc-lift stress 1.0` | Умный пол. Поднимает передний план, защищая дальние планы и небо. | <img src="docs/showcase/03_auto_tgc_stress.png" width="200"/> |
| **Advanced**<br>`--auto --tgc-lift stress 1.2 --soft-floor 2.5` | Монотонный soft-floor сохраняет градиенты. Генерирует карту рисков. | <img src="docs/showcase/04_advanced_prepress.png" width="200"/> |
| **Pro-Tip**<br>`--auto --tgc-lift stress 1.0 --black-anchor 0.03` | Фиксирует истинный черный (Dmax). Генерирует QA-лист. | <img src="docs/showcase/05_black_anchor_qa.png" width="200"/> |

---

### 🚀 Быстрый старт

#### Требования и установка

```bash
pip install -r requirements.txt
```

**Подготовка нейросетевых моделей:**

```bash
python check_models.py --download
```

Скрипт скачает веса MiDaS в папку `./models/`, а движок автоматически верифицирует контрольную сумму SHA256.

---

### 🛠 Использование (CLI)

**1. Полный автомат со Smart Preserve (Рекомендуется для колод Таро и карт):**

```bash
python shadow_lift_prepress.py -i ./cards_raw/ -o ./cards_out/ -s 900x1500 --auto --smart-preserve --tgc-lift stress
```

**2. Точный контроль параметров (Ручной режим):**

```bash
python shadow_lift_prepress.py -i ./input.png -o ./output.png -s 900x1500 -b 17 --tgc-lift stress 0.7 --soft-floor 2.8 --black-anchor 0.055 --smart-preserve
```

**3. Пакетная параллельная обработка:**

```bash
python shadow_lift_prepress.py -i ./cards_raw/ -o ./cards_out/ -s 900x1500 --auto --smart-preserve --workers 4
```

**4. Генерация QA-листов контроля качества (включая 4-панельный лист из showcase):**

```bash
python shadow_lift_prepress.py -i ./cards_raw/ -o ./cards_out/ --auto --qa-dir ./qa_reports/
```

---

### ⚠️ КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Параметр `--cmyk-compression`

Параметр `--cmyk-compression` предназначен **ИСКЛЮЧИТЕЛЬНО ДЛЯ ПРЕВЬЮ**. Его задача — эвристически показать на экране монитора возможный сдвиг насыщенности.

* 🚫 **НЕ используйте `--cmyk-compression > 0` для мастер-файлов, отправляемых в печать!**
* Если типография применяет свой ICC-профиль (например, TGC или GRACoL), использование этого параметра приведет к **двойной цветокоррекции** и испортит тени на отпечатке. Для печатных файлов всегда оставляйте `0.0`.

---

### 📋 Основные аргументы командной строки

| Аргумент | Флаг | По умолчанию | Описание |
| --- | --- | --- | --- |
| `--input` | `-i` | *Обязательный* | Путь к исходному файлу или папке. |
| `--output` | `-o` | *Обязательный* | Путь для сохранения (папка или путь `.png`). |
| `--size` | `-s` | `None` | Целевой размер `ШИРИНАxВЫСОТА` (например, `900x1500`). |
| `--auto` |  | `False` | Автономная настройка параметров на основе непрерывного анализа сцены. |
| `--smart-preserve` |  | `False` | **v1.7.4.** Семантический защитник Dark Scene (пространственный Sky Prior, предотвращение швов и дыма). |
| `--tgc-lift` |  | `off` | `MODE [STRENGTH]` — `off` / `safe` / `stress`. Компенсация усадки принтера. |
| `--soft-floor` |  | `0.0` | Мягкий монотонный floor (0–20 RGB). Заменяет жесткую отсечку. |
| `--black-anchor` |  | `0.0` | Фиксация истинного черного (Dmax) в диапазоне 0.015–0.055. |
| `--deband-strength` |  | `auto` | Сила дизеринга и устранения бандинга (0.0–2.0). |
| `--qa-dir` |  | `None` | Папка для генерации 4-панельных контактных листов качества. |
| `--prepress-filter` |  | `False` | Генерация файлов симуляции печати и карт риска оверкриппинга (`*_risk.png`). |
| `--workers` |  | `1` | Количество параллельных процессов обработки. |
| `--accept-models` |  | `False` | Пропускать запрос подтверждения при расхождении SHA256 весов MiDaS. |
| `--cmyk-compression` |  | `0.0` | ⚠️ ТОЛЬКО ПРЕВЬЮ. Эвристическая десатурация теней. |
| `--debug` |  | `False` | Сохранение отладочных карт масок для разработчиков. |

---

## ⚠️ ДИСКЛЕЙМЕР И ОТКАЗ ОТ ОТВЕТСТВЕННОСТИ

**Данный проект представляет собой экспериментальный набор инструментов для предварительной обработки изображений. Он НЕ является сертифицированным решением для подготовки материалов к печати и не дает никаких коммерческих гарантий.**

Результаты могут различаться. Предварительный просмотр на экране и симулятор фильтров TGC не позволяют в полной мере предсказать, как изображение будет выглядеть на физическом отпечатке. **Перед запуском полномасштабной печати обязательно проверяйте итоговые настройки с помощью пробного оттиска.**

**ИСПОЛЬЗУЯ ЭТОТ СКРИПТ, ВЫ ДЕЛАЕТЕ ЭТО ИСКЛЮЧИТЕЛЬНО НА СВОЙ СТРАХ И РИСК.** Автор(ы) не несет ответственности за любой прямой или косвенный ущерб, включая, но не ограничиваясь: испорченные файлы, брак при печати, финансовые потери, упущенную выгоду или любые другие последствия, возникшие в результате использования данного программного обеспечения.

---

## 🇬🇧 Shadow-Lift Depth/Shadow Prepress Engine

**An experimental Python toolkit (v1.7.4 — Intelligent Arcana & Smart-Preserve Edition) for automated prepress processing of complex low-key images, photographs, and artwork intended for digital and offset printing.**

The toolkit focuses on shadow detail visibility and atmosphere preservation through adaptive tonal correction in Oklab color space, spatially guided processing (MiDaS depth, structure tensor, guided filter), semantic Dark Scene Analysis, and **three-mode printer compensation for The Game Crafter (TGC)**.

---

### 📐 Project Structure

| File / Folder | Purpose | Dependencies |
| --- | --- | --- |
| `shadow_lift_prepress.py` | Core prepress engine v1.7.4 | torch, cv2, scipy, PIL, numpy |
| `dark_scene/` | Semantic Dark Scene detection module (`--smart-preserve`) | cv2, scipy, numpy |
| `dark_scene/darkness_intent.py` | Regional Intent Analyzer with Sky Prior & Spatial Awareness | numpy |
| `dark_scene/mask_generator.py` | Continuous soft-mask generator for protected voids | cv2, scipy, numpy |
| `check_models.py` | Model inspector, SHA256 verifier, and weight downloader | standard library only |
| `calibrate.py` | TGC curve calibration from physical print | PIL + numpy only |
| `default_curve.json` | TGC simulator parameters (auto-loaded) | — |
| `requirements.txt` | Python dependencies | — |

---

### 🖼 Result Showcase

Example of the 4-panel QA sheet generated with the `--qa-dir` flag. From left to right: input image, MiDaS depth map, shadow map, and final prepress output.

| Input Image | Depth Map (MiDaS) | Shadow Map | Prepress Output |
| :---: | :---: | :---: | :---: |
| <img src="docs/showcase_input.jpg" width="200"/> | <img src="docs/showcase/showcase_depth.png" width="200"/> | <img src="docs/showcase_shadow.png" width="200"/> | <img src="docs/showcase_output.jpg" width="200"/> |

*(Note: replace `docs/...` paths with the actual filenames in your repository).*

---

### 🧠 Core Features & v1.7.4 Updates

* **Dark Scene Analyzer v2.1 (`--smart-preserve`):** Intent analysis engine with **Sky Prior** (spatial awareness for the upper frame hemisphere). Protects deep void, space, and night skies from unnatural lifting or banding.
* **Patch 1 (Ground Rescue):** Uncouples `target_map` from `depth_protection` in `RECOVER` zones to fully restore foreground elements.
* **Patch 2 (Smoke Killer):** Neutralizes TGC hard floor and RGB offsets smoothly within `PRESERVE` zones, eliminating "gray smoke" haze on physical prints.
* **Oklab Continuous Brightness Gate:** Lifting and TGC floor compensation are controlled via a smooth continuous brightness gate along the Oklab L-channel ($\sigma=15.0$). This eliminates sharp boundary artifacts ("sky seams") between background skies and adjacent architecture.
* **MiDaS Depth Model:** Intel ISL `midas_v21_small_256.pt` with automatic SHA256 verification.
* **Continuous Auto-Tuner (`--auto`):** Autonomous scene archetype classifier tailored for low-key illustration, occult metals, fire hotspots, and night scenes.

---

### 🚀 Quick Start

```bash
pip install -r requirements.txt
python check_models.py --download
python shadow_lift_prepress.py -i ./cards_raw/ -o ./cards_out/ -s 900x1500 --auto --smart-preserve --tgc-lift stress
```

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE).
```