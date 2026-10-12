"""Ukrainian wording for the engine-generated table titles, headers and cells."""
from __future__ import annotations

import re

EXACT = {
    # headers
    "Basis": "Базис", "c_b": "c_б", "step": "крок", "node": "вершина", "distance": "відстань",
    "previous": "попередня", "rank": "місце", "method": "метод", "edge": "ланка", "length": "довжина",
    "action": "дія", "consumer": "споживач", "sweep order": "порядок обходу", "angle, °": "кут, °",
    "route": "маршрут", "state": "стан", "vertex": "вершина", "constraint": "обмеження", "used": "використано",
    "limit": "ліміт", "remainder": "залишок", "dist. to initial": "до початкового", "dist. to refined": "до уточненого",
    "from": "з", "to": "у", "allocated": "вкладено", "profit": "приріст", "cumulative": "накопичено",
    "stage": "етап", "cells": "клітинок", "iterations": "ітерацій",
    # cells
    "added": "додано", "skipped (cycle)": "пропущено (контур)",
    # method names
    "North-west corner": "Північно-західного кута", "North-east corner": "Північно-східного кута",
    "South-west corner": "Південно-західного кута", "South-east corner": "Південно-східного кута",
    "Minimum cost by column": "Мінімальної вартості за стовпцем",
    "Minimum cost by row": "Мінімальної вартості за рядком",
    "Minimum cost in table": "Мінімальної вартості за таблицею",
    "Double preference": "Подвійної переваги", "Vogel approximation": "Фогеля",
    # titles
    "Vertices of the feasible region": "Вершини області допустимих розв’язків",
    "Constraints": "Обмеження", "Kruskal's algorithm": "Алгоритм Краскала",
    "Sweep method": "Метод Свіра", "Distances": "Відстані",
    "Permanent labels (order of fixing)": "Послідовність присвоєння постійних міток",
    "Potential plans sorted by cost": "Потенційні плани, відсортовані за витратами",
    "Optimal plan": "Оптимальний план", "All transitions": "Усі переходи",
    "Best cumulative profit of each state": "Найкращий сумарний приріст для кожного стану",
    "Simplex: optimal tableau": "Симплекс-метод: оптимальна таблиця",
    "Phase I: optimal tableau": "Фаза I: оптимальна таблиця",
}

PATTERNS = [
    (re.compile(r"^Simplex: iteration (\d+) — (\S+) enters, (\S+) leaves$"),
     lambda m: "Симплекс-метод: ітерація {} – вводиться {}, виводиться {}".format(m[1], m[2], m[3])),
    (re.compile(r"^Phase I: iteration (\d+) — (\S+) enters, (\S+) leaves$"),
     lambda m: "Фаза I: ітерація {} – вводиться {}, виводиться {}".format(m[1], m[2], m[3])),
    (re.compile(r"^Cut (\d+) added \(row (\S+)\)$"),
     lambda m: "Симплекс-таблиця з доданим відтинанням {} (рядок {})".format(m[1], m[2])),
    (re.compile(r"^Cut (\d+): dual simplex step (\d+) — (\S+) leaves, (\S+) enters$"),
     lambda m: "Відтинання {}: крок {} двоїстого симплекс-методу – виводиться {}, вводиться {}".format(
         m[1], m[2], m[3], m[4])),
    (re.compile(r"^Tableau after cut (\d+)$"), lambda m: "Симплекс-таблиця після відтинання {}".format(m[1])),
    (re.compile(r"^Stage (\d+)$"), lambda m: "Етап {} (підприємство № {})".format(m[1], m[1])),
    (re.compile(r"^x(\d+)\*$"), lambda m: "x{}*".format(m[1])),
    (re.compile(r"^stage (\d+)$"), lambda m: "етап {}".format(m[1])),
    (re.compile(r"^Cut (\d+): variable (\S+) = (\S+) has the largest fractional part (\S+) → (.+)$"),
     lambda m: "Відтинання {}: змінна {} = {} має найбільшу дробову частину {}; додаткове обмеження {}"
     .format(m[1], m[2], m[3], m[4], m[5])),
]


def uk(text) -> str:
    """Translate one engine string; anything unknown is returned unchanged."""
    if not isinstance(text, str):
        return text
    value = text.strip()
    if value in EXACT:
        return EXACT[value]
    for pattern, repl in PATTERNS:
        m = pattern.match(value)
        if m:
            return repl(m)
    return text
