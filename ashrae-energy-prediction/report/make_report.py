"""PDF report (ReportLab). Run from the project root after report/make_figures.py:

    python report/make_report.py            -> report/ashrae_energy_report.pdf
"""
import math
import os
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("Body", FONTS / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Bold", FONTS / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold")

styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=styles["BodyText"], fontName="Body", fontSize=9.2,
                      leading=13, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", fontName="Bold", fontSize=17, leading=21, spaceAfter=8, textColor=colors.HexColor("#1f3b5c"))
H2 = ParagraphStyle("h2", fontName="Bold", fontSize=12, leading=15, spaceBefore=8, spaceAfter=5, textColor=colors.HexColor("#1f3b5c"))
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.8, leading=10, alignment=0)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#555555"), alignment=0)
CELL = ParagraphStyle("cell", fontName="Body", fontSize=7.4, leading=9)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Bold")
WIDTH = A4[0] - 3.6 * cm


def para(text, style=BODY):
    return Paragraph(text, style)


def figure(name, caption, width=WIDTH):
    path = Path("results/figures") / name
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return [Image(str(path), width=width, height=width * h / w), para(caption, CAPTION), Spacer(1, 6)]


def table(frame, widths, header_rows=1, bold_rows=()):
    data = [[Paragraph(str(c), CELLB) for c in frame.columns]] + [
        [Paragraph(str(v), CELLB if i in bold_rows else CELL) for v in row]
        for i, row in enumerate(frame.itertuples(index=False))
    ]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f2")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c4d0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


comparison = pd.read_csv("results/model_comparison.csv")
submissions = pd.read_csv("results/kaggle_submissions.csv").sort_values("date")
board = pd.read_csv("results/ashrae-energy-prediction-privateleaderboard-2026-10-13.csv")
teams = len(board)
scores = board["Score"].to_numpy()
position = lambda s: int((scores < s).sum()) + 1

story = []

# --- Страница 1: постановка и итог ---------------------------------------------
story += [
    para("Прогноз потребления энергии зданиями: ASHRAE — Great Energy Predictor III", H1),
    para("Отчёт по проекту: постановка, данные, признаки, сравнение моделей, результаты на Kaggle", CAPTION),
    Spacer(1, 6),
    para("Задача", H2),
    para("Соревнование <b>ASHRAE — Great Energy Predictor III</b> (Kaggle): по почасовым показаниям 2016 года у 1 449 зданий "
         "на 16 площадках и по погоде спрогнозировать показания четырёх типов счётчиков (электричество, холодная вода, пар, "
         "горячая вода) на следующие два года, 2017–2018. В тесте 41.7 млн строк при 20.2 млн строк обучения. "
         "Метрика — RMSLE, то есть RMSE от <i>log1p(meter_reading)</i>; чем меньше, тем лучше."),
    para("Главные трудности: прогноз на период, которого нет в обучении (только один год наблюдений, тест — следующие два); "
         "сбои счётчиков в обучающих метках; 8 ГБ памяти на ноутбуке и 41.7 млн строк теста."),
    para("Итог", H2),
]
best = submissions.dropna(subset=["privateScore"]).sort_values("privateScore").iloc[0]
summary = pd.DataFrame([
    ["Лучший Private Score", f"{best['privateScore']:.3f} (позиция {position(best['privateScore'])} из {teams}, Top {position(best['privateScore']) / teams * 100:.0f}%)"],
    ["Лучший Public Score", f"{best['publicScore']:.3f}"],
    ["Старт (первая отправка)", f"Private 1.428, позиция {position(1.428)}, Top {position(1.428) / teams * 100:.0f}%"],
    ["Порог Top 20%", f"{board.iloc[math.ceil(teams * 0.2) - 1]['Score']:.3f} (позиция {math.ceil(teams * 0.2)})"],
    ["Финальная модель", "Стэк из LightGBM, CatBoost и MLP, валидация блоками месяцев: RMSE 1.047"],
    ["Время полного прогона", "около 1 ч валидация блоками с подбором, около 1.9 ч прогноз теста (MacBook M1, 8 ГБ)"],
], columns=["Показатель", "Значение"])
story += [table(summary, [5.0 * cm, WIDTH - 5.0 * cm]), Spacer(1, 8)]
story += [
    para("Что сделано", H2),
    para("От линейного бейслайна (RMSE 1.99) через общие и локальные бустинги, подбор параметров и нейросеть — к стэку, "
         "обучаемому на трёх блоках месяцев. Улучшения проверялись измерениями на валидации, а не принимались на веру: "
         "часть идей (окна температуры, праздники, удаление постоянных серий) показала почти нулевой эффект и в отчёте "
         "описана именно так. Результаты Kaggle (история отправок и итоговый leaderboard) загружаются через Kaggle CLI."),
]
story += figure("horizon.png", "Рис. 1. Ошибка растёт вместе с удалением от обучающих данных: на проверке внутри 2016 года "
                "результат оптимистичнее, чем на тесте.", width=WIDTH * 0.62)
story.append(PageBreak())

# --- Страница 2: данные ---------------------------------------------------------
story += [para("1. Данные и их особенности", H1)]
story += [
    para("Таблицы: показания счётчиков (<i>train</i>, <i>test</i>), метаданные зданий (площадь, год постройки, этажность, назначение) "
         "и почасовая погода площадок. Время погоды приведено к местному смещением по площадкам — без этого суточный пик "
         "температуры не совпадает с пиком потребления."),
    para("Электричество составляет 60% строк, но тепловые счётчики (холодная вода, пар, горячая вода) определяют ошибку: "
         "они сильно зависят от сезона и режима здания (рис. 2, справа). Распределение показаний тяжёлое, поэтому и "
         "модели, и метрика работают в логарифмической шкале."),
]
story += figure("data_overview.png", "Рис. 2. Доли строк по счётчикам, распределение log1p(meter_reading), сезонный профиль 2016 года.")
story += [
    para("Сбои в обучающих метках", H2),
    para("Известный дефект: на площадке 0 электричество до 20 мая нулевое — эти строки исключены из обучения. "
         "Дополнительно исключены серии нулей электричества и постоянных значений длиной от недели (2.8% строк при пороге 168 часов, "
         "6.1% при 24 часах, рис. 3). Нули пара и воды не удаляются: летом они закономерны и повторятся в тесте."),
]
story += figure("cleaning.png", "Рис. 3. Исключённые серии (красные точки) и доля исключённых строк по счётчикам.")
story.append(PageBreak())

# --- Страница 3: признаки и протокол --------------------------------------------
story += [para("2. Признаки и протокол проверки", H1)]
story += [
    para("Признаки", H2),
    para("<b>Основа:</b> десять полиномиальных комбинаций (площадь, её степени, произведения с температурой, точкой росы, "
         "облачностью, давлением, ветром), циклические час и направление ветра, коды <i>meter</i>, <i>site_id</i>, "
         "<i>primary_use</i>, <i>building_id</i>. <b>Добавлено:</b> сырые погодные и строительные значения с пропусками "
         "(деревья обрабатывают NaN сами), час и день недели, среднее температуры за 3, 24 и 72 часа, температура сутки "
         "назад, отклонение от среднего за 72 часа (всё по площадке и только по прошлому), праздники по странам площадок, месяц."),
    para("Протокол", H2),
    para("Первая схема — «январь–июль → август → сентябрь»: оценивает только следующий месяц и даёт оптимистичные 1.00. "
         "Итоговая — три блока по 4 месяца: модель учится на двух блоках и прогнозирует третий, то есть сезон, которого не видела. "
         "Третий блок делится случайно пополам: на первой половине (<i>meta</i>) выбираются параметры и обучается метамодель, "
         "на второй (<i>predict</i>) считается итоговая ошибка. Все модели в таблице ниже считались по этой схеме."),
]
story += figure("ablations.png", "Рис. 4. Слева: вклад признаков (общая LightGBM, блоки месяцев). Справа: пороги чистки. "
                "Месяц помогает заметно; окна температуры и праздники — почти нет; удаление постоянных серий отдельно не помогает.")
story += [
    para("Что дали признаки и чистка (общая LightGBM, блоки месяцев)", H2),
    para("Месяц снижает ошибку на 0.008 (100 деревьев) и на 0.036 (400 деревьев, 127 листьев); окна температуры и праздники — в пределах "
         "±0.003; день года хуже месяца. Чистка нулей электричества улучшает RMSE с 1.139 до 1.118; добавление постоянных серий "
         "ухудшает до 1.124, но оставлено по условию задачи. Эффект месяца — оценка по одному году; будет ли он таким на 2017–2018, "
         "на Kaggle ещё не проверено."),
]
story.append(PageBreak())

# --- Страница 4-5: сравнение моделей --------------------------------------------
story += [para("3. Сравнение моделей", H1)]
cmp = comparison.copy()
cmp.insert(0, "№", range(1, len(cmp) + 1))
table_data = pd.DataFrame({
    "№": cmp["№"],
    "Модель": cmp["model"],
    "Класс": cmp["family"].map({"linear": "линейная", "boosting, global": "бустинг, общая", "boosting, local": "бустинг, локальные",
                                "NN": "нейросеть", "ensemble": "ансамбль"}),
    "RMSE": cmp["RMSE"].map("{:.3f}".format),
    "блок 0": cmp["block 0"].map("{:.3f}".format),
    "блок 1": cmp["block 1"].map("{:.3f}".format),
    "блок 2": cmp["block 2"].map("{:.3f}".format),
    "Время, мин": cmp["train_minutes"].map("{:.1f}".format),
})
story += [
    para("RMSE на log1p по половине <i>predict</i> трёх блоков; время — обучение и прогноз суммарно по трём блокам. "
         "Время стэка — сумма времён входящих моделей (метамодель мгновенная); подбор параметров в него не входит "
         "(LightGBM 25 мин, CatBoost 36 мин).", SMALL),
    Spacer(1, 4),
    table(table_data, [0.9 * cm, 6.6 * cm, 2.9 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.5 * cm], bold_rows=(0, 1)),
    Spacer(1, 8),
]
story += figure("model_comparison.png", "Рис. 5. Ошибка по моделям и качество против времени обучения (номера как в таблице).")
story.append(PageBreak())

# --- Страница 6: анализ ------------------------------------------------------------
story += [para("4. Анализ: почему модели различаются", H1)]
analysis = [
    ("Линейная модель против деревьев (1.84–1.99 против 1.06–1.12).",
     "Потребление — произведение площади, разности температур и графика работы по часам; линейная модель выражает только "
     "сумму. One-Hot по зданию даёт каждому свой уровень (1.99 → 1.84), но взаимодействия остаются недоступны. "
     "Время 0.1 мин не окупает ошибки в 1.7 раза выше."),
    ("Общая модель против локальных (1.122 → 1.063 → 1.075 → 1.115).",
     "Разделение по <i>meter</i> снижает ошибку на 0.06: четыре вида счётчиков по-разному зависят от погоды. Более мелкие группы "
     "хуже: на здание приходится около 420 строк против 250 тыс. на счётчик, и модель здания, видевшая 8 месяцев, "
     "плохо экстраполирует на сезон, которого не видела. Причина — объяснение по числу строк, отдельно не проверялась."),
    ("LightGBM против CatBoost (1.075 против 1.112).",
     "CatBoost хуже при любой группировке и в 3–8 раз медленнее. Измерено: большая модель (глубина 10, 600 итераций) хуже "
     "маленькой (глубина 4, 400 итераций) — 1.126 против 1.112, то есть CatBoost переобучается. Общая CatBoost со 100 итерациями "
     "(1.270) недообучена на миллионе строк."),
    ("Подбор параметров почти ничего не даёт.",
     "LightGBM: 1.075 → 1.072 ценой 13-кратного времени; разброс между блоками у одной модели (1.05–1.09) больше этого эффекта. "
     "У CatBoost выигрыш 0.03 достигнут в основном уменьшением модели."),
    ("MLP слабее деревьев (1.156), но получает вес 0.25.",
     "Корреляция её ошибок с ошибками деревьев 0.77–0.84, между деревьями 0.87–0.97: сеть даёт гладкий ответ, деревья — "
     "кусочно-постоянный, и на незнакомом сезоне они ошибаются по-разному."),
    ("Стэк 1.047 против лучшей одиночной 1.063.",
     "Простое среднее пяти моделей даёт 1.049, Ridge — 1.047: выигрыш даёт разнообразие моделей, а не подбор весов. "
     "Самая сильная одиночная модель (LightGBM по <i>meter</i>) кандидатом стэка не была — первое, что стоит попробовать."),
]
for head, text in analysis:
    story += [para(f"<b>{head}</b> {text}")]
story += figure("errors.png", "Рис. 6. Слева: RMSE по типу счётчика. Справа: корреляция ошибок моделей стэка.")
story.append(PageBreak())

# --- Страница 7: Kaggle -----------------------------------------------------------
story += [para("5. Результаты на Kaggle", H1)]
notes = {
    57122102: "Стэк 5 моделей + MLP (вес 0.54), вес по августу",
    57136288: "Стэк 4 модели без LightGBM по зданиям",
    57141061: "Сырая погода, час, чистка, окно 72 ч, клиппинг",
    57147426: "Повторная отправка предыдущего файла",
}
sub = submissions.copy()
kaggle_table = pd.DataFrame({
    "Дата (UTC)": sub["date"].str.slice(0, 16),
    "Что внутри": sub["ref"].map(notes),
    "Public": sub["publicScore"].map("{:.3f}".format),
    "Private": sub["privateScore"].map("{:.3f}".format),
    "Позиция": [f"{position(s)} ({position(s) / teams * 100:.0f}%)" for s in sub["privateScore"]],
})
thresholds = pd.DataFrame([
    {"Процентиль": f"Top {p}%", "Позиция": math.ceil(teams * p / 100), "Private Score": f"{board.iloc[math.ceil(teams * p / 100) - 1]['Score']:.3f}"}
    for p in (1, 5, 10, 20, 25, 50)
])
story += [
    para("Результаты выгружаются из Kaggle CLI (<i>kaggle competitions submissions</i>, <i>kaggle competitions leaderboard</i>); "
         f"положение считается по итоговому private leaderboard ({teams} команд).", BODY),
    table(kaggle_table, [3.1 * cm, 7.2 * cm, 1.6 * cm, 1.6 * cm, 2.6 * cm]),
    Spacer(1, 6),
    table(thresholds, [3.2 * cm, 3.2 * cm, 3.2 * cm]),
    Spacer(1, 8),
]
story += figure("leaderboard.png", "Рис. 7. Распределение private score команд и наши отправки (красные линии), порог Top 20% (зелёная пунктирная).")
story += [
    para("Прирост 1.428 → 1.338 (на 0.09) дали несколько изменений одновременно; по отдельности они на Kaggle не мерились. "
         "До порога Top 20% (1.295) остаётся 0.043. Итоговая версия с месяцем, праздниками и валидацией блоками "
         "на момент отчёта ещё не отправлена."),
]
story.append(PageBreak())

# --- Страница 8: выводы -------------------------------------------------------------
story += [para("6. Выводы и ограничения", H1)]
story += [
    para("Выводы", H2),
    para("1. Главная трудность — не точность подгонки, а перенос на другие годы: проверка внутри 2016 года завышает качество "
         "(1.00–1.05 против 1.34 на private), поэтому её значения годятся для сравнения моделей, но не как прогноз результата."),
    para("2. Дешёвые решения оказались не хуже дорогих: LightGBM по типу счётчика за 0.4 мин лучше всех одиночных моделей; "
         "подбор параметров (около часа) и CatBoost глубины 10 не окупились."),
    para("3. Разнообразие ансамбля важнее весов: простое среднее почти не уступает Ridge. Добавлять надо модели другого типа."),
    para("4. Ошибка сосредоточена на тепловых счётчиках (две трети квадратичной ошибки при 41% строк)."),
    para("Ограничения и риски", H2),
    para("• Один год обучения: эффект месяца и праздников не проверяем без второго года. "
         "• Отдельный вклад признаков на Kaggle не измерен (изменения отправлялись пакетом). "
         "• Часть объяснений (число строк на модель, структура деревьев CatBoost) — предположения, прямо не проверялись. "
         "• Общие модели по <i>meter</i> в тестовой схеме (площадка за площадкой) пока не реализованы. "
         "• Прогноз ограничен максимумом train: 9 строк выше, в основном здания 791–803 на площадке 7 около 30.10.2017."),
    para("Что делать дальше", H2),
    para("Добавить в стэк общие LightGBM по типу счётчика (обучить один раз на выборке, прогнозировать по площадкам), "
         "убрать CatBoost глубины 10, добавить модель другого типа (например, нормированная на площадь цель), "
         "отправить итоговую версию и сопоставить результат с оценкой на блоках."),
    Spacer(1, 10),
    para("Воспроизведение: см. README.md. Ноутбук <i>research.ipynb</i>, данные и ключи Kaggle в репозиторий не входят.", SMALL),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(1.8 * cm, 1.1 * cm, "ASHRAE — Great Energy Predictor III · отчёт")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"стр. {doc.page}")
    canvas.restoreState()


out = Path("report/ashrae_energy_report.pdf")
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                  topMargin=1.6 * cm, bottomMargin=1.7 * cm,
                  title="ASHRAE Great Energy Predictor III: отчёт", author="Denis").build(
    story, onFirstPage=footer, onLaterPages=footer)
print("written", out, os.path.getsize(out) // 1024, "KB")
