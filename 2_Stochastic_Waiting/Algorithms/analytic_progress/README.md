# Analytic Progress Module

Dieses Paket trennt die analytische Routing-Logik in klaren Schichten:

- `models.py`: Eingabemodell (`WarehouseInstance`) mit Validierung
- `core.py`: reine Routenberechnungen (`T_in`, `T_out`, `T_end`, Positionen, V/R-Mengen)
- `metrics.py`: erweiterbare Zusatzmetriken (z. B. Umwege)
- `io_cli.py`: CLI-Argumente und Demo-Instanz
- `main.py`: Orchestrierung als Einstiegspunkt

## Schnellstart

```powershell
python -m Algorithms.analytic_progress.main
python -m Algorithms.analytic_progress.main --times 10 20 30 40
```

