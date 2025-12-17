# game-env-analyzer

## Описание проекта

Основная цель — построение и оценка архитектуры, способной:

1. Детектировать врагов на кадрах игрового процесса.
2. Классифицировать тип врага.
3. Определять текущий паттерн атаки врага на основе визуального контекста.

## Постановка задачи

* **Вход**: RGB-кадры игрового процесса (screen capture).
* **Выход**:

  * Bounding box каждого врага на кадре.
  * Класс врага.
  * Класс атаки, выполняемой врагом в данный момент времени.

Особенностью задачи является:

* высокая внутриклассовая вариативность атак,
* перекрытие персонажей и эффектов,
* быстро меняющаяся анимация,
* отсутствие готовых размеченных датасетов.

## Подход и архитектура

### Общий пайплайн

Приняли решение использовать **двухэтапный подход**:

1. **Object Detection + Enemy Classification**
2. **Attack Pattern Classification**

```
Input Frame
     ↓
YOLO (Enemy Detection + Class)
     ↓
Crop Bounding Boxes
     ↓
ResNet (Attack Pattern Classification)
```

### Детекция врагов

Для детекции используется модель семейства **YOLO**, обученная на пользовательском датасете.

**Задачи модели:**

* локализация врагов (bounding boxes),
* классификация типа врага.

### Классификация паттернов атак

#### CustomLightNet

Для классификации паттернов атак используется **легковесная сверточная нейронная сеть с skip connection**, вдохновлённая архитектурой ResNet, но адаптированная под ограниченный размер датасета и требования к скорости инференса.

Модель реализована в виде класса `CustomLightNet` и предназначена для работы с кропами bounding box’ов врагов, полученными на этапе детекции.

#### ResidualBlock

Базовым элементом сети является `ResidualBlock`, реализующий стандартный residual-подход:

* две 3×3 свертки,
* Batch Normalization после каждой свертки,
* нелинейность ReLU,
* skip-connection с возможностью изменения размерности.

#### Общая структура сети

Архитектура сети имеет иерархическую структуру с постепенным увеличением количества каналов и уменьшением spatial-размерности:

| Stage      | Channels | Stride | Blocks                |
| ---------- | -------- | ------ | --------------------- |
| Input Conv | 16       | 1      | Conv + BN + LeakyReLU |
| Stage 1    | 16       | 1      | 2 × ResidualBlock     |
| Stage 2    | 32       | 2      | 2 × ResidualBlock     |
| Stage 3    | 64       | 2      | 2 × ResidualBlock     |
| Stage 4    | 128      | 2      | 2 × ResidualBlock     |
| Head       | 128 → C  | –      | GAP + FC              |

#### Выход модели

Модель решает задачу **multiclass classification**.

* Вход: `Tensor[B, 3, H, W]`
* Выход: `Tensor[B, num_classes]`
* Функция потерь: CrossEntropyLoss


## Датасет

### Сбор данных

Датасет был собран самостоятельно из записей игрового процесса *Hollow Knight*.

Процесс:

1. Запись видеоматериала игрового процесса.
2. Извлечение кадров.
3. Ручная разметка данных в **CVAT**.

### Разметка

Разметка включает:

* bounding box врага,
* класс врага,
* класс атаки (на уровне кадра).

### Структура датасета


```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── yolo_format/
│   └── attack_labels/
└── annotations.xml
```

### Содержание датасета
Для данной работы были выбраны следующие противники: _Hornet_, _Crystal Guard_ и _False Knight_.
Ниже представлены примеры данных и статистика датасета для каждого противника.

<table>
  <tr>
    <td colspan="5" align="center">
      <h3>Hornet</h3>
      <p><em>Состояния противника Hornet</em></p>
    </td>
  </tr>
  <tr align="center">
    <td><img src="https://github.com/user-attachments/assets/bc2e885d-f547-48da-86cd-e468aa92b304" width="120"></td>
    <td><img src="https://github.com/user-attachments/assets/d9ac54e7-e225-4f1a-8eba-380816107f57" width="120"></td>
    <td><img src="https://github.com/user-attachments/assets/2fd8b84f-4efb-4a25-bbc9-7eaa0c9de177" width="120"></td>
    <td><img src="https://github.com/user-attachments/assets/30ea0f42-87f2-40db-91ad-78cdd5e2d06b" width="120"></td>
    <td><img src="https://github.com/user-attachments/assets/e9c1c18e-5c28-4c40-a06e-9b549718b363" width="120"></td>
  </tr>
  <tr align="center">
    <td><b>Не производит атаку</b><br>на игрока</td>
    <td>Атака <b>hornet_ram</b></td>
    <td>Атака <b>hornet_throw</b></td>
    <td>Атака <b>hornet_drill</b></td>
    <td>Атака <b>hornet_silk</b></td>
  </tr>
  <tr align="center" style="border-top: 1px solid #ddd;">
    <td><small>Количество примеров: 1496</small></td>
    <td><small>Количество примеров: 170</small></td>
    <td><small>Количество примеров: 146</small></td>
    <td><small>Количество примеров: 219</small></td>
    <td><small>Количество примеров: 64</small></td>
  </tr>
</table>

<table>
  <tr>
    <td colspan="4" align="center">
      <h3>Crystal Guard</h3>
      <p><em>Состояния противника Crystal Guard</em></p>
    </td>
  </tr>
  <tr align="center">
    <td><img src="https://github.com/user-attachments/assets/000ed20c-cdb6-441f-a5e2-1e88609e9994" width="120"></td>
    <td><img src="https://github.com/user-attachments/assets/5657817b-ed1b-4c0d-b3e7-86019e83580b" width="120"></td>
    <td><img src="https://github.com/user-attachments/assets/94b46e7c-bb71-40ff-af6f-2be8150a7399" width="120"></td>
    <td><img src="https://github.com/user-attachments/assets/c660f24f-7688-44f3-ad3c-a4fc6e033077" width="120"></td>
  </tr>
  <tr align="center">
    <td><b>Не производит атаку</b><br>на игрока</td>
    <td>Атака <b>Scream and beams</b></td>
    <td>Атака <b>Jump</b></td>
    <td>Атака <b>Hand laser</b></td>
  </tr>
  <tr align="center" style="border-top: 1px solid #ddd;">
    <td><small>Количество примеров: 517</small></td>
    <td><small>Количество примеров: 222</small></td>
    <td><small>Количество примеров: 261</small></td>
    <td><small>Количество примеров: 453</small></td>
  </tr>
</table>

<table>
  <tr>
    <td colspan="5" align="center">
      <h3>False Knight</h3>
      <p><em>Состояния противника False Knight</em></p>
    </td>
  </tr>
  <tr align="center">
    <td><img src="https://github.com/user-attachments/assets/401b2a0d-b0f1-4584-8a61-034c6e8c6443" width="120"></td>
    <td><img src="https://github.com/user-attachments/assets/3436d56d-b2d0-4ce1-9bdb-1f3b18364956" width="120"></td>
    <td><img src="https://github.com/user-attachments/assets/df7de093-6b39-421e-8f3c-f6f6d9ca23e5" width="120"></td>
    <td><img src="https://github.com/user-attachments/assets/4bd155dc-0fbe-40a9-b049-729e6294896b" width="120"></td>
    <td><img src="https://github.com/user-attachments/assets/454726d6-6422-4846-af67-c6d46368571f" width="120"></td>
  </tr>
  <tr align="center">
    <td><b>Не производит атаку</b><br>на игрока</td>
    <td>Атака <b>Throwing wave</b></td>
    <td>Атака <b>Hit from heaven</b></td>
    <td>Атака <b>Left-right smash</b></td>
    <td>Атака <b>Jump</b></td>
  </tr>
  <tr align="center" style="border-top: 1px solid #ddd;">
    <td><small>Количество примеров: 1172</small></td>
    <td><small>Количество примеров: 264</small></td>
    <td><small>Количество примеров: 469</small></td>
    <td><small>Количество примеров: 256</small></td>
    <td><small>Количество примеров: 346</small></td>
  </tr>
</table>

## Эксперименты и метрики
Результаты:  
|             | precision  |  recall | f1-score  | support
|-------------|------------|---------|-----------|--------
|      hornet |    0.9703  |  0.9833 |   0.9767  |     299
|hornet_drill |    0.8750  |  0.9655 |   0.9180  |      29
|  hornet_ram |    1.0000  |  0.8824 |   0.9375  |      34
| hornet_silk |    1.0000  |  0.9231 |   0.9600  |      13
|hornet_throw |    0.9762  |  0.9318 |   0.9535  |      44
|-------------|------------|---------|-----------|--------
|    accuracy |            |         |   0.9666  |     419
|   macro avg |    0.9643  |  0.9372 |   0.9492  |     419
|weighted avg |    0.9677  |  0.9666 |   0.9665  |     419

Пока обучались только на одном боссфайте, но в репозитории есть интерфейс для объединения датасетов с разных боссфайтов в один и экстраполяции модели  
<img width="1113" height="237" alt="image" src="https://github.com/user-attachments/assets/69d7be61-4e1a-4dd5-8878-375175956950" />  
  
В `inference.ipynb` реализована live-версия модели, вот скриншоты её работы:  
<img width="949" height="548" alt="image_2025-12-17_06-27-11" src="https://github.com/user-attachments/assets/f615922c-4486-471a-b2a9-ba879a3a2bc6" />
<img width="891" height="437" alt="image_2025-12-17_06-40-44" src="https://github.com/user-attachments/assets/d3bf917e-bef8-4a81-88ed-4424abf7b10f" />

