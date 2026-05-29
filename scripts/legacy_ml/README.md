# scripts/legacy_ml/

Pipeline **ML historico** (scripts 02 a 16): descarga, scoring, feature store,
entrenamiento de modelos v1/v2/v3, backtests y comparaciones de versiones.

**NO** forma parte del flujo activo bajo Plan C. Se archiva el 28/5/2026 como:
- **receta reproducible** de como se entrenaron/compararon los modelos, util para
  futuras comparaciones (modelo viejo vs nuevo) y reuso de logica de features/scoring.
- forma de **ordenar** `scripts/` (separar lo historico del flujo diario activo).

No se borran (tienen valor de opcion para comparaciones futuras y costo casi nulo).
El historial completo igual vive en git.
