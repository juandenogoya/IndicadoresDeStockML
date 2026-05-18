"""
Tools de catalogo de queries (Fase 2).

Escribe SOLO en ~/queries-catalog/, repo Git separado.
Nunca toca el repo principal ni la DB.

Tools (pendientes de implementar):
  save_query(name, sql, description, tags) -> dict
      Guarda query en ~/queries-catalog/catalog/<tag>/<name>.sql
      con metadata en .md adyacente.
      Annotation: readOnlyHint=False (requiere confirmacion del usuario).

  list_saved_queries(tag) -> list[dict]
      Lista queries guardadas con descripcion y tags.

  recall_query(name) -> dict
      Devuelve SQL + metadata de una query guardada.
"""
