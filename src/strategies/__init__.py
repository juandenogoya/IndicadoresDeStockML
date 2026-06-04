"""
src/strategies/ -- Logica de decision COMPARTIDA entre Forward Testing (local)
y los bots Alpaca paper (Plan B, Tarea 16).

Principio (Opcion B): un solo "cerebro". Estas funciones son PURAS: reciben
data (en la forma de los dicts que calzan con las columnas de senales_bot_diaria)
y devuelven decisiones (a_abrir / a_cerrar). No tocan la DB ni el broker.

  - El ACCESO A DATOS lo hace cada lado (FT lee DB local; Alpaca lee la masticada).
  - La DECISION vive aca (compartida) -> FT y Alpaca deciden identico (homologacion).
  - La EJECUCION la hace cada lado (FT escribe ft_*; Alpaca llama alpaca_client).

Modulos:
  scoring.py     -> calcular_score_tecnico (SMA/MACD/RSI, 3 capas)
  sectorial.py   -> evaluar_entradas_sectorial / evaluar_cierres_sectorial (v1 y v2)
  ml_scanner.py  -> evaluar_entradas_ml / evaluar_cierres_ml
"""
