"""
Validacion SQL con sqlglot -- segunda linea de defensa.

Recibe un string SQL y verifica que sea exclusivamente SELECT o WITH...SELECT.
Rechaza DML/DDL y funciones de filesystem peligrosas de PostgreSQL.
Inyecta LIMIT automaticamente si la query no trae uno.

Funciones publicas:
  validate_select(sql: str) -> str
      Retorna el SQL (posiblemente con LIMIT inyectado) o lanza SafetyError.

Excepciones:
  SafetyError(Exception) -- SQL rechazado, con razon en el mensaje.
"""
