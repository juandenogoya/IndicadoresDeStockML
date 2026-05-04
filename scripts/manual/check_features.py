import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env.local"), override=True)
except ImportError:
    pass

from src.data.database import query_df

r1 = query_df("SELECT MAX(fecha) as ult, COUNT(*) as filas FROM features_market_structure")
r2 = query_df("SELECT MAX(fecha) as ult, COUNT(*) as filas FROM features_precio_accion")

print()
print(f"  features_market_structure : ultima={r1.iloc[0]['ult']}  filas={r1.iloc[0]['filas']:,}")
print(f"  features_precio_accion    : ultima={r2.iloc[0]['ult']}  filas={r2.iloc[0]['filas']:,}")
print()
