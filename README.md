# Boryscad 2

Математический редактор в стиле Mathcad на Python: свободное размещение регионов на листе, живой пересчёт, символьные и численные вычисления, блоки решения `given … find/maximize/minimize`, графики, изображения, настраиваемые темы — и набор функций исследования операций, которым автоматизируются все 9 лабораторных работ из «122б ДО 2023».

## Установка и запуск

```bash
pip install -e F:\Python\Boryscad2
boryscad                       # графический редактор
boryscad gui lab3.bmath        # открыть файл
```

Без установки: `python -m boryscad`.

## Терминал

```bash
boryscad render boryscad/examples/lab3_nonlinear.bmath            # → lab3_nonlinear.html
boryscad render "boryscad/examples/*.bmath" -o out -f pdf -t Paper
boryscad formula "x := √(a^2 + b^2)" -o formula.png
boryscad run boryscad/examples/lab7_gomory.bmath -q                # результаты прямо в терминал
boryscad convert lab3.bmath lab3.bcad
boryscad themes
boryscad functions "Operations"
```

Форматы экспорта: HTML (формулы — векторный SVG), PDF (A4, постранично), PNG, SVG.

## Язык формул

| Запись | Смысл |
|---|---|
| `x := 5`, `f(x, y) := x^2 + y` | определения |
| `f(2, 3) =` | вычисление |
| `(x+1)^2 -> expand` | символьно: `expand`, `factor`, `simplify`, `solve(x)`, `diff(x)`, `integrate(x)`, `series(x, 0, 5)`, `float(6)`, `substitute(x = 2)`, `laplace(t, s)`… |
| `i := 0..10`, `t := 0..1 step 0.1`, `t := 0, 0.1 .. 1` | диапазоны |
| `v[i] := i^2` | векторизованное присваивание (`ORIGIN := 1` — нумерация с 1) |
| `A := [1, 2; 3, 4]`, `A^T`, `A^-1`, `|A|`, `A[0, 1]` | матрицы |
| `given` … `find(x, y)` / `maximize(f, x, y)` / `minimize(f, x, y)` / `minerr(x, y)` | блоки решения |
| `deriv(f(x), x)`, `integral(f(x), x, a, b)`, `sum(k^2, k, 1, n)`, `limit(…)` | анализ (рисуются как d/dx, ∫, Σ, lim) |
| `g(n) := { s <- 0; for k in 1..n { s <- s + k }; s }` | программы: `for`, `while`, `if/else`, `return`, `break` |
| `r := simplex(c, A, b)`, `r.x =` | записи с полями |

Максимизация/минимизация выбирает стратегию сама: точное ЛП (HiGHS, в т.ч. `integer=1`), точный метод Лагранжа/KKT для малых полиномиальных задач и численный SLSQP с мультистартом.

## Файл `.bmath`

```
@title ЛР 3
# Заголовок
> Абзац с **Markdown** и подстановкой значения {{x}}.
N := 5
f(x) := x^2 - N

@plot title="f" x=-5..5 width=520 height=320
fplot(f(t), t, -5, 5, label="f")
@end
@image picture.png width=300 caption="Рис. 1"
@row / @end        — регионы рядом
@pagebreak
```

## Лабораторные работы

| ЛР | Пример | Функции |
|---|---|---|
| 1. Графический метод ЛП | `lab1_graphical_lp.bmath` | `lp_graphical`, `lp_graph`, `given/maximize` |
| 2. Симплекс-метод | `lab2_simplex.bmath` | `simplex` (все таблицы), `linprog`, `randmatrix`, `seed` |
| 3. Нелинейное программирование | `lab3_nonlinear.bmath` | `given/minimize/maximize`, `contour`, `region`, `implicit` |
| 4. Транспортная задача | `lab4_transport.bmath` | `transport_all` (9 методов, сортировка, потенциалы), `transport` |
| 5. Коммивояжёр | `lab5_tsp.bmath` | `mst`, `shortest_paths`, `route_count`, `sweep`, `tsp`, `graph` |
| 6. Положение склада | `lab6_location.bmath` | `weber`, `centroid`, `warehouse`, `sumdist` |
| 7. Метод Гомори | `lab7_gomory.bmath` | `gomory` (все симплекс-таблицы и отсечения) |
| 8. Алгоритм Дейкстры | `lab8_dijkstra.bmath` | `dijkstra`, `graph(path=…)` |
| 9. Распределение ресурсов | `lab9_allocation.bmath` | `allocate` (Беллман), `allocate_graph` (стоимость перехода) |

В GUI: *Файл → Примеры (лабораторные)*. Меняйте `N`, `k` — всё пересчитывается.

## Архитектура

```
boryscad/
  language/   grammar.lark (LALR), ast.py, parser.py        — синтаксис
  engine/     evaluator, operators, environment, symbolic (SymPy),
              registry (реестр функций), worksheet (вычисление документа)
  library/    elementary, linalg, calculus, statistics, symbolic, solving,
              plotting, optimization, transport, graphs, integer, dynamic, location
  core/       document (модель), bmath/io (форматы), theme, markdown
  render/     boxes (вёрстка формул), builder (AST → boxes), qt_backend,
              plot_renderer (matplotlib), regions (виды регионов), document (экспорт)
  ui/         main_window, worksheet (QGraphicsScene), editor, panels, dialogs,
              controller (undo/redo), calc (фоновый пересчёт), style (QSS + иконки)
```

Точки расширения (Open/Closed):

* **Функции** — `FunctionRegistry.function(...)`; плагины: любой `~/.boryscad/plugins/*.py` с `register(registry)`.
* **Символьные ключевые слова** — `registry.symbolic_command`.
* **Типы регионов** — `RegionFactory.register`, `WorksheetEngine.register_region_evaluator`, `render.regions.REGION_VIEWS`.
* **Серии графиков** — `render.plot_renderer.series_painter("kind")`.
* **Стратегии решения** — классы `Strategy` в `library/solving.py`.
* **Темы** — JSON в `boryscad/themes` или `~/.boryscad/themes` (редактор цветов и шрифтов в *Вид → Настройки и стиль*).

Движок не зависит от GUI, рендеринг — от конкретного бэкенда (протоколы `Measurer`/`Canvas`).

## Горячие клавиши

Щелчок по листу + ввод — новая формула; `"` — текст; Enter — готово; Shift+Enter — новая строка; двойной щелчок / F2 — редактировать; Del — удалить; стрелки — сдвиг; Ctrl+колесо — масштаб; F9 — пересчёт; Ctrl+Z/Y — отмена/повтор; Ctrl+C/V — копирование регионов и вставка картинок из буфера; перетаскивание файлов изображений на лист.

## Тесты

```bash
python -m pytest tests
```
