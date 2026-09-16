# Batch Combination Optimizer (Calculate_Intervals)

Dieses Skript berechnet fuer **alle Pick-Kombinationen** (Multinomial-Verteilung) ueber die Gassen:

- Intervalle via `interval_builder.build_interval_data`
- optimales `wait*` durch numerische Minimierung von `J(wait)` (Python)
- kritisches `delta*` durch Root-Suche auf `J'(0)=0` (Python)
- gewichtete Erwartungswerte mit den Auftretenswahrscheinlichkeiten
- XLSX-Export (`summary` + `details`)

Datei: `batch_combination_optimizer.py`

## Standard-Run

```powershell
python -u "C:\Users\ya5909\Git-Projekte\project_4D4L\2_Stochastic_Waiting\Algorithms\analytic_progress\Calculate_Intervals\batch_combination_optimizer.py"
```

## Schneller Testlauf (Teilmenge)

```powershell
python -u "C:\Users\ya5909\Git-Projekte\project_4D4L\2_Stochastic_Waiting\Algorithms\analytic_progress\Calculate_Intervals\batch_combination_optimizer.py" --n-picks 2 --max-combos 10
```

## Wichtige Parameter

- `--delta-exp`: Delta fuer die `wait*`-Optimierung (default `28.8`)
- `--n-picks`: Anzahl Picks fuer die Kombinationserzeugung (default `3`)
- `--n-picks-after`: Parameter fuer `ECostAfter` (default `3`)
- `--output`: Zielpfad fuer die XLSX-Datei
- `--max-combos`: optional nur erste N Kombinationen (nach Wahrscheinlichkeit)

## Hinweis

Fuer XLSX-Export wird `openpyxl` benoetigt. Falls es fehlt:

```powershell
pip install openpyxl
```

