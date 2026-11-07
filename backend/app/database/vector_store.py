"""
pgvector_store.py - a vector store class for PostgreSQL + pgvector.

SETUP
    pip install "psycopg[binary]" sentence-transformers

QUICK START
    from embedding_service import EmbeddingService

    embeddings = EmbeddingService.from_model_name("sentence-transformers/all-MiniLM-L6-v2")
    with VectorStore("host=localhost dbname=postgres user=postgres password=pw",
                     embeddings=embeddings, collection="docs") as store:
        store.create_collection()
        store.upsert_documents(["Reset your password in Settings."],
                               metadatas=[{"source": "help.md"}])
        store.create_index("hnsw")          # build the index AFTER loading data
        print(store.similarity_search("how do I change my password?", k=3))

A "collection" is a Postgres table:  id | content | metadata (JSONB) | embedding | created_at
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ..embeddings.embedding_service import EmbeddingService



logger = logging.getLogger(__name__)

# distance name -> (SQL operator, index operator class)
_DISTANCES = {
    "cosine": ("<=>", "vector_cosine_ops"),
    "l2": ("<->", "vector_l2_ops"),
    "ip": ("<#>", "vector_ip_ops"),  # note: <#> returns the NEGATIVE inner product
}


class VectorStore:
    """Store and search embeddings in PostgreSQL with pgvector."""

    def __init__(
        self,
        dsn: str,
        embeddings: EmbeddingService,
        collection: str,
        dimension: Optional[int] = None,
    ) -> None:
        """
        dsn        : Postgres connection string
        embeddings : an EmbeddingService instance (it owns the model)
        collection : default collection (table) name
        dimension  : embedding size; taken from the embedding service if omitted
        """
        if embeddings is None:
            raise ValueError("embeddings must not be None.")
        self.embeddings = embeddings
        self.collection = self._validate_name(collection)
        self.dimension = dimension or embeddings.dimension
        if self.dimension < 1:
            raise ValueError("dimension must be a positive integer")

        # autocommit: each statement commits itself; conn.transaction() is used where atomicity is needed
        if conninfo_to_dict(dsn).get("host") == "localhost":
            dsn = make_conninfo(dsn, host="127.0.0.1")
        self.conn = psycopg.connect(
            dsn,
            connect_timeout=10,
            autocommit=True,
            row_factory=dict_row,
        )

    # =========== connection and setup ==============

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def health_check(self) -> Dict[str, Any]:
        """Confirm the database is reachable and report the pgvector version."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
        return {
            "connected": True,
            "pgvector_version": row["extversion"] if row else "not installed",
        }

    # ============= Helpers ===================

    @staticmethod
    def _validate_name(name: str) -> str:
        # Table/index names can't be SQL parameters, so only allow safe characters.
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", name or ""):
            raise ValueError(
                "name must be lowercase letters, digits or underscores "
                "(max 63 chars, no leading digit)"
            )
        return name

    def _table(self, collection: Optional[str]) -> sql.Identifier:
        """Turn a collection name into a safely quoted SQL identifier."""
        return sql.Identifier(self._validate_name(collection or self.collection))

    @staticmethod
    def _vec(vector: List[float]) -> str:
        """Format a Python list as a pgvector literal, e.g. '[0.1,0.2,0.3]'."""
        return "[" + ",".join(repr(float(x)) for x in vector) + "]"

    @staticmethod
    def _where(where: Optional[dict]):
        """Metadata filter using JSONB containment (speeded up by a GIN index)."""
        if not where:
            return sql.SQL(""), []
        return sql.SQL(" WHERE metadata @> %s"), [Jsonb(where)]

    # ============= Collections ==============

    def create_collection(self, name: Optional[str] = None) -> None:
        """Create a collection (table). Does nothing if it already exists."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id         TEXT PRIMARY KEY,
                        content    TEXT NOT NULL,
                        metadata   JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding  vector({dim}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                ).format(table=self._table(name), dim=sql.SQL(str(int(self.dimension))))
            )
        logger.info("Collection '%s' ready (dimension=%d)", name or self.collection, self.dimension)

    def collection_exists(self, name: Optional[str] = None) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass(%s) IS NOT NULL AS ok",
                (self._validate_name(name or self.collection),),
            )
            return cur.fetchone()["ok"]

    def list_collections(self) -> List[str]:
        """All public tables that have a vector column called 'embedding'."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name FROM information_schema.columns
                WHERE table_schema = 'public' AND column_name = 'embedding'
                  AND udt_name IN ('vector', 'halfvec') ORDER BY table_name
                """
            )
            return [r["table_name"] for r in cur.fetchall()]

    def delete_collection(self, name: Optional[str] = None, confirm: bool = False) -> None:
        """Drop a collection and ALL its data. Requires confirm=True."""
        if not confirm:
            raise ValueError("delete_collection permanently deletes data; pass confirm=True to proceed")
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(self._table(name)))

    def list_indexes(self, name: Optional[str] = None) -> List[str]:
        """Names of the indexes on a collection."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s "
                "ORDER BY indexname",
                (self._validate_name(name or self.collection),),
            )
            return [r["indexname"] for r in cur.fetchall()]

    def get_collection_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Row count, vector column type/dimension and existing indexes."""
        name = self._validate_name(name or self.collection)
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
                "WHERE attrelid = to_regclass(%s) AND attname = 'embedding'",
                (name,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"collection '{name}' does not exist")
            cur.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(self._table(name)))
            n = cur.fetchone()["n"]
        return {"name": name, "rows": n, "vector_type": row["t"], "indexes": self.list_indexes(name)}

    def create_index(
        self,
        kind: str = "hnsw",
        distance: str = "cosine",
        collection: Optional[str] = None,
    ) -> None:
        """Create an ANN index (hnsw or ivfflat) plus a GIN index on metadata.
        Build AFTER loading data, and use the same distance in similarity_search."""
        if kind not in ("hnsw", "ivfflat"):
            raise ValueError("kind must be 'hnsw' or 'ivfflat'")
        if distance not in _DISTANCES:
            raise ValueError(f"distance must be one of {list(_DISTANCES)}")
        table_name = self._validate_name(collection or self.collection)
        opclass = _DISTANCES[distance][1]
        vec_idx = self._validate_name(f"{table_name}_{kind}_{distance}"[:59] + "_idx")
        meta_idx = self._validate_name(f"{table_name}"[:55] + "_meta_idx")

        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} USING {kind} (embedding {opclass})").format(
                    idx=sql.Identifier(vec_idx),
                    table=sql.Identifier(table_name),
                    kind=sql.SQL(kind),
                    opclass=sql.SQL(opclass),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} USING gin (metadata)").format(
                    idx=sql.Identifier(meta_idx), table=sql.Identifier(table_name)
                )
            )
        logger.info("Indexes ready on '%s' (%s, %s)", table_name, kind, distance)

    # ============= Writing data =====================

    def _write(self, texts, ids, metadatas, collection, upsert: bool) -> List[str]:
        if not texts:
            raise ValueError("texts is empty")
        ids = [str(i) for i in ids] if ids else [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]
        if not (len(ids) == len(metadatas) == len(texts)):
            raise ValueError("texts, ids and metadatas must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("ids contain duplicates")

        vectors = self.embeddings.generate_embeddings(texts)  # embed first, so a failure writes nothing
        conflict = (
            sql.SQL(
                " ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, "
                "metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding"
            )
            if upsert
            else sql.SQL("")
        )
        query = sql.SQL(
            "INSERT INTO {table} (id, content, metadata, embedding) VALUES (%s, %s, %s, %s::vector){conflict}"
        ).format(table=self._table(collection), conflict=conflict)
        rows = [(i, t, Jsonb(m), self._vec(v)) for i, t, m, v in zip(ids, texts, metadatas, vectors)]
        with self.conn.transaction(), self.conn.cursor() as cur:  # all rows or none
            cur.executemany(query, rows)
        return ids

    def add_documents(self, texts: List[str], ids: Optional[List[str]] = None,
                      metadatas: Optional[List[dict]] = None, collection: Optional[str] = None) -> List[str]:
        """Embed and INSERT chunks. Fails if an id already exists. Returns the ids used."""
        return self._write(texts, ids, metadatas, collection, upsert=False)

    def upsert_documents(self, texts: List[str], ids: Optional[List[str]] = None,
                         metadatas: Optional[List[dict]] = None, collection: Optional[str] = None) -> List[str]:
        """Embed and INSERT new chunks, or UPDATE existing ones with the same id."""
        return self._write(texts, ids, metadatas, collection, upsert=True)

    def delete_documents(self, ids: Optional[List[str]] = None, where: Optional[dict] = None,
                         collection: Optional[str] = None) -> int:
        """Delete by ids and/or metadata filter. Refuses to run with neither."""
        if not ids and not where:
            raise ValueError("provide ids and/or where; use delete_collection to remove everything")
        conds, params = [], []
        if ids:
            conds.append(sql.SQL("id = ANY(%s)"))
            params.append([str(i) for i in ids])
        if where:
            conds.append(sql.SQL("metadata @> %s"))
            params.append(Jsonb(where))
        query = sql.SQL("DELETE FROM {table} WHERE ").format(table=self._table(collection)) + sql.SQL(
            " AND "
        ).join(conds)
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return cur.rowcount

    def update_metadata(self, id: str, metadata: dict, merge: bool = True,
                        collection: Optional[str] = None) -> bool:
        """Change a chunk's metadata without re-embedding. merge=False replaces it."""
        expr = sql.SQL("metadata || %s") if merge else sql.SQL("%s")
        query = sql.SQL("UPDATE {table} SET metadata = {expr} WHERE id = %s").format(
            table=self._table(collection), expr=expr
        )
        with self.conn.cursor() as cur:
            cur.execute(query, (Jsonb(metadata), str(id)))
            return cur.rowcount > 0

    # ============= Searching =====================

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        where: Optional[dict] = None,
        distance: str = "cosine",
        collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the k nearest chunks to the query (smaller distance = closer)."""
        if k < 1:
            raise ValueError("k must be >= 1")
        if distance not in _DISTANCES:
            raise ValueError(f"distance must be one of {list(_DISTANCES)}")

        op = sql.SQL(_DISTANCES[distance][0])
        qvec = self._vec(self.embeddings.generate_query_embedding(query))
        where_sql, where_params = self._where(where)
        statement = sql.SQL(
            "SELECT id, content, metadata, embedding {op} %s::vector AS distance "
            "FROM {table}{where} ORDER BY embedding {op} %s::vector LIMIT %s"
        ).format(op=op, table=self._table(collection), where=where_sql)

        with self.conn.cursor() as cur:
            cur.execute(statement, [qvec, *where_params, qvec, k])
            return cur.fetchall()

        