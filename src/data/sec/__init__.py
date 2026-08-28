"""
src/data/sec -- ingesta de la fuente SEC XBRL.

REGLA DE DEPENDENCIA (una sola direccion, no relajarla):
    nada de este paquete importa del lado de trading del proyecto
    (src/scoring, src/trading, src/strategies, src/pipeline, scanner, bots).
    Solo stdlib, requests, y el normalizador PURO src/utils/sec_xbrl.py.

    El motivo es que la costura quede cortada de antemano: si algun dia
    aparece un segundo consumidor de esta fuente, `git subtree split` sobre
    src/data/sec + src/utils/sec_xbrl.py + tests/test_sec_xbrl.py la separa
    sin desenredar nada. Si alguna vez hace falta importar del lado de
    trading, es senal de que algo esta mal ubicado.

    La orquestacion (universo, DB, UPSERT) vive en
    scripts/refresh_fundamentales_sec.py, NO aca.
"""
