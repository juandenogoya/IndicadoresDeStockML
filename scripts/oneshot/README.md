# scripts/oneshot/

Scripts de **una sola vez** ya ejecutados: migraciones, DDL de creacion de tablas
y limpiezas puntuales. **No** forman parte del flujo diario. Se conservan como
registro historico y por si hay que recrear una tabla.

Movidos aca el 28/5/2026 para ordenar `scripts/migrations/`.

Quedaron en `scripts/migrations/` (activos / reutilizables, NO one-shot):
- `sync_railway_to_local.py`  (lo usa `scripts/manual/sync_local.bat`)
- `sync_local_to_railway.py`  (lo usa `scripts/manual/sync_to_railway.bat`)
- `clean_ticker_fantasma_se.py`  (limpieza generica reutilizable)
