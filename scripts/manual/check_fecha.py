"""
check_fecha.py
Verifica si una fecha es dia habil NYSE y muestra el contexto.

Uso:
    python scripts/manual/check_fecha.py                 # hoy
    python scripts/manual/check_fecha.py 2026-04-27      # fecha especifica
    python scripts/manual/check_fecha.py --rango 5       # proximos 5 dias habiles

Exit codes:
    0  dia habil
    1  fin de semana o feriado (util para scripts .bat que necesiten validar)
"""

import os
import sys
import argparse
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.trading_calendar import (
    is_trading_day, holiday_name, prev_trading_day,
    next_trading_day, trading_days_between, describe_date, _WEEKDAY_ES
)


def main():
    parser = argparse.ArgumentParser(description="Verificador de dias habiles NYSE")
    parser.add_argument("fecha", nargs="?", help="Fecha YYYY-MM-DD (default: hoy)")
    parser.add_argument("--rango", type=int, default=0,
                        help="Mostrar N dias habiles siguientes (default: 0)")
    args = parser.parse_args()

    # Parsear fecha
    if args.fecha:
        try:
            d = date.fromisoformat(args.fecha)
        except ValueError:
            print(f"\n  ERROR: formato invalido '{args.fecha}'. Usar YYYY-MM-DD\n")
            sys.exit(2)
    else:
        d = date.today()

    habil = is_trading_day(d)
    festivo = holiday_name(d)
    dow = _WEEKDAY_ES[d.weekday()]

    print()
    print("=" * 56)
    print(f"  CHECK FECHA NYSE  |  consultado: {date.today()}")
    print("=" * 56)
    print()
    print(f"  Fecha    : {d}  ({dow})")

    if habil:
        print(f"  Estado   : DIA HABIL NYSE")
    elif festivo:
        print(f"  Estado   : FERIADO NYSE  ({festivo})")
    elif d.weekday() >= 5:
        nombre_fin = "Sabado" if d.weekday() == 5 else "Domingo"
        print(f"  Estado   : FIN DE SEMANA  ({nombre_fin})")

    print()
    print(f"  Dia habil anterior : {prev_trading_day(d)}")
    print(f"  Dia habil siguiente: {next_trading_day(d)}")

    # Mostrar rango de dias habiles proximos
    if args.rango > 0:
        print()
        print(f"  Proximos {args.rango} dias habiles:")
        siguiente = next_trading_day(d) if not habil else d
        conteo = 0
        cur = siguiente if not habil else d
        while conteo < args.rango:
            if is_trading_day(cur):
                festivo_cur = holiday_name(cur)
                nota = f"  <- FERIADO: {festivo_cur}" if festivo_cur else ""
                print(f"    {cur}  {_WEEKDAY_ES[cur.weekday()]}{nota}")
                conteo += 1
            cur += timedelta(days=1)

    print()
    print("=" * 56)
    print()

    # Exit code util para .bat
    sys.exit(0 if habil else 1)


if __name__ == "__main__":
    main()
