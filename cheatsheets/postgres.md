# Postgres Cheat Sheet

As of Nov 1, 2026. Editable version: [Postgres Cheat Sheet doc](https://claude.ai/artifact/1CLzJUrgZiTSYcytnUadp8)

## Mental model and psql

Postgres is a disk-based, ACID relational database that runs one OS process per connection and uses MVCC so readers never block writers.

| Idea | What it means | Consequence |
| --- | --- | --- |
| Process per connection | Each client gets a backend process (~5-10 MB) | Use a pooler (PgBouncer) beyond a few hundred connections |
| MVCC | Updates write a new row version; old versions stay until VACUUM | Readers do not block writers; tables bloat without vacuum |
| WAL | Every change is logged before data pages are written | Crash recovery, replication and point-in-time restore all come from WAL |
| Shared buffers + OS cache | Pages are 8 KB, cached twice | Set `shared_buffers` ~25% of RAM |
| Cost-based planner | Chooses plans from table statistics | Stale stats (no `ANALYZE`) cause bad plans |

```bash
docker run -d --name pg -e POSTGRES_PASSWORD=pw -p 5432:5432 postgres:17
docker exec -it pg psql -U postgres
```

| psql | Does |
| --- | --- |
| `\l` / `\c db` | List databases / connect |
| `\dt` / `\d table` / `\di` | List tables / describe table / list indexes |
| `\dn` / `\du` | Schemas / roles |
| `\x auto` | Expanded output for wide rows |
| `\timing on` | Show query time |
| `\e` | Edit query in `$EDITOR` |
| `\i file.sql` | Run a file |
| `\copy t FROM 'f.csv' CSV HEADER` | Client-side CSV import |

## SQL essentials

These snippets cover most daily work: schema, upsert, joins, CTEs, window functions and JSONB.

```sql
-- Schema
CREATE TABLE users (
  id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  email      text NOT NULL UNIQUE,
  meta       jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE orders (
  id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  amount  numeric(12,2) NOT NULL CHECK (amount >= 0),
  status  text NOT NULL DEFAULT 'new'
);
ALTER TABLE orders ADD COLUMN note text;          -- instant, no rewrite

-- Upsert
INSERT INTO users (email) VALUES ('a@x.com')
ON CONFLICT (email) DO UPDATE SET meta = users.meta || '{"seen":true}'
RETURNING id;

-- Join + aggregate
SELECT u.email, count(o.id) AS orders, coalesce(sum(o.amount),0) AS total
FROM users u LEFT JOIN orders o ON o.user_id = u.id
GROUP BY u.email HAVING count(o.id) > 1
ORDER BY total DESC LIMIT 10;

-- CTE + window: each user's latest order
WITH ranked AS (
  SELECT o.*, row_number() OVER (PARTITION BY user_id ORDER BY id DESC) AS rn
  FROM orders o
)
SELECT * FROM ranked WHERE rn = 1;

-- Running total
SELECT id, amount, sum(amount) OVER (ORDER BY id) AS running FROM orders;

-- Recursive CTE (tree walk)
WITH RECURSIVE t AS (
  SELECT id, parent_id FROM nodes WHERE id = 1
  UNION ALL
  SELECT n.id, n.parent_id FROM nodes n JOIN t ON n.parent_id = t.id
)
SELECT * FROM t;

-- JSONB
SELECT meta->>'plan' FROM users WHERE meta @> '{"plan":"pro"}';
UPDATE users SET meta = jsonb_set(meta, '{plan}', '"free"') WHERE id = 1;

-- Keyset pagination (faster than OFFSET)
SELECT * FROM orders WHERE id < $1 ORDER BY id DESC LIMIT 50;
```

| Prefer | Over | Why |
| --- | --- | --- |
| `timestamptz` | `timestamp` | Stores UTC, avoids time zone bugs |
| `text` | `varchar(n)` | Same speed; use `CHECK` for length |
| `bigint` identity | `serial` | Standard SQL, no 2^31 overflow |
| `jsonb` | `json` | Indexable, deduplicated keys |
| `numeric` | `float` | Exact money math |

## Indexes and EXPLAIN

Match the index type to the operator in your `WHERE` clause, and always confirm with `EXPLAIN (ANALYZE, BUFFERS)`.

| Index | Good for | Example |
| --- | --- | --- |
| B-tree (default) | `=`, `<`, `>`, `BETWEEN`, `ORDER BY`, prefix `LIKE 'abc%'` | `CREATE INDEX ON orders (user_id, id DESC);` |
| Composite | Multi-column filters; leftmost columns must be used | `(status, created_at)` serves `status = ?` and `status = ? AND created_at > ?` |
| Partial | Hot subset of rows | `CREATE INDEX ON orders (id) WHERE status = 'new';` |
| Covering (`INCLUDE`) | Index-only scans | `CREATE INDEX ON orders (user_id) INCLUDE (amount);` |
| Expression | Functions in filters | `CREATE INDEX ON users (lower(email));` |
| GIN | JSONB `@>`, arrays, full-text, trigram `LIKE '%x%'` | `CREATE INDEX ON users USING gin (meta jsonb_path_ops);` |
| GiST / SP-GiST | Geometry, ranges, nearest-neighbour | PostGIS, `tstzrange` exclusion constraints |
| BRIN | Huge append-only tables ordered by time | `CREATE INDEX ON events USING brin (created_at);` |
| HNSW / IVFFlat (pgvector) | Vector similarity search | `USING hnsw (embedding vector_cosine_ops)` |

In production, build with `CREATE INDEX CONCURRENTLY` so writes are not blocked.

```sql
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM orders WHERE user_id = 42;
```

| Plan node | Meaning | Red flag |
| --- | --- | --- |
| Seq Scan | Reads the whole table | On a big table with a selective filter: missing index |
| Index Scan / Index Only Scan | Uses index (only-scan skips the heap) | `Heap Fetches` high: run VACUUM |
| Bitmap Heap Scan | Many matching rows, combines indexes | Fine for medium selectivity |
| Nested Loop | Per-row lookup into inner side | Huge outer row count |
| Hash Join / Merge Join | Bulk joins | `Batches > 1`: raise `work_mem` |
| Sort (external merge) | Sort spilled to disk | Raise `work_mem` or add an index |
| rows=estimate vs actual | Planner guess vs reality | 10x+ mismatch: `ANALYZE`, extended statistics |

## Transactions, MVCC and locking

The default isolation is Read Committed; use Serializable when correctness depends on what other transactions read, and retry on error `40001`.

| Isolation level | Prevents | Allows | Note |
| --- | --- | --- | --- |
| Read Committed (default) | Dirty reads | Non-repeatable reads, phantoms, lost updates | Each statement sees a fresh snapshot |
| Repeatable Read | + non-repeatable reads, phantoms | Write skew | One snapshot per transaction; may fail with serialization error |
| Serializable (SSI) | All anomalies | Nothing | Retry on `40001`; some overhead |

```sql
BEGIN;
SELECT balance FROM accounts WHERE id = 1 FOR UPDATE;   -- row lock
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;

-- Job queue: each worker grabs a different row
SELECT id FROM jobs WHERE status = 'new'
ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED;

-- App-level mutex
SELECT pg_advisory_xact_lock(42);

-- Guard migrations
SET lock_timeout = '5s';
SET statement_timeout = '30s';
```

| Lock / tool | Use | Watch out |
| --- | --- | --- |
| `FOR UPDATE` | Lock rows you will modify | Lock rows in a consistent order to avoid deadlocks |
| `FOR UPDATE SKIP LOCKED` | Queues with many workers | Results may be partial by design |
| `FOR SHARE` | Block changes while reading | Rarely needed |
| Advisory locks | Custom mutexes (cron leader) | Session-level locks leak with poolers in transaction mode |
| `ALTER TABLE` | Takes `ACCESS EXCLUSIVE` | Queues behind long queries and blocks everyone: set `lock_timeout` |

**MVCC and VACUUM**

- Each row version has `xmin` / `xmax` transaction ids; `UPDATE` = new version + old marked dead.
- `VACUUM` reclaims dead rows for reuse; `VACUUM FULL` rewrites the table with an exclusive lock (prefer `pg_repack`).
- Autovacuum also prevents transaction-ID wraparound; never disable it, tune it per table instead.
- Long-running transactions and idle-in-transaction sessions stop vacuum from cleaning: set `idle_in_transaction_session_timeout`.
- HOT updates avoid index writes when no indexed column changes and the page has free space (`fillfactor`).

## Replication, scaling and backups

Scale reads with streaming replicas, scale connections with PgBouncer, and split big tables with declarative partitioning before reaching for sharding (Citus).

| Feature | How it works | Key facts |
| --- | --- | --- |
| Streaming replication | Replica replays primary's WAL | Async by default; `synchronous_commit = on` + `synchronous_standby_names` for zero loss |
| Logical replication | Publish/subscribe of row changes per table | Cross-version upgrades, CDC (Debezium), selective tables |
| Failover / HA | Patroni (with etcd), or the CloudNativePG operator on Kubernetes | Postgres has no built-in automatic failover |
| Replication slots | Primary keeps WAL until consumer reads it | Abandoned slot fills the disk: monitor `pg_replication_slots` |
| PgBouncer | Pools many client connections onto few backends | Transaction mode breaks session features (`SET`, advisory session locks, `LISTEN`) |
| Partitioning | `PARTITION BY RANGE / LIST / HASH` | Partition pruning; drop old data with `DROP` / `DETACH PARTITION` instead of `DELETE` |
| Citus | Distributed tables across nodes | Shard key choice decides join locality |

```sql
CREATE TABLE events (id bigint, created_at timestamptz NOT NULL, payload jsonb)
  PARTITION BY RANGE (created_at);
CREATE TABLE events_2026_11 PARTITION OF events
  FOR VALUES FROM ('2026-11-01') TO ('2026-12-01');
```

| Backup | Type | Restore point |
| --- | --- | --- |
| `pg_dump -Fc db > db.dump` / `pg_restore` | Logical, one database | Time of dump |
| `pg_basebackup` | Physical, whole cluster | Time of backup |
| pgBackRest / WAL-G + WAL archiving | Physical + continuous WAL | Any point in time (PITR) |

Test restores regularly; an untested backup is not a backup.

## Tuning and observability

Start every investigation with `pg_stat_statements`; it ranks queries by total time.

| Setting | Starting point | Why |
| --- | --- | --- |
| `shared_buffers` | 25% of RAM | Postgres page cache |
| `effective_cache_size` | 50-75% of RAM | Planner hint for OS cache |
| `work_mem` | 16-64 MB | Per sort/hash node, per query; too high risks OOM |
| `maintenance_work_mem` | 512 MB - 2 GB | Faster VACUUM and index builds |
| `max_connections` | 100-300 + PgBouncer | Each connection is a process |
| `random_page_cost` | 1.1 on SSD | Default 4 assumes spinning disks |
| `max_wal_size` | 4-16 GB | Fewer checkpoints under write load |

```sql
-- Top queries by total time (CREATE EXTENSION pg_stat_statements;)
SELECT calls, round(total_exec_time) AS ms, round(mean_exec_time,2) AS avg_ms, query
FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 10;

-- What is running / waiting right now
SELECT pid, state, wait_event_type, wait_event, now() - query_start AS age, query
FROM pg_stat_activity WHERE state <> 'idle' ORDER BY age DESC;

-- Who blocks whom
SELECT pid, pg_blocking_pids(pid) AS blocked_by, query
FROM pg_stat_activity WHERE cardinality(pg_blocking_pids(pid)) > 0;

-- Kill a query / session
SELECT pg_cancel_backend(12345);  SELECT pg_terminate_backend(12345);

-- Table sizes
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_statio_user_tables ORDER BY pg_total_relation_size(relid) DESC LIMIT 10;

-- Unused indexes
SELECT relname, indexrelname, idx_scan FROM pg_stat_user_indexes WHERE idx_scan = 0;

-- Dead tuples and last autovacuum
SELECT relname, n_dead_tup, last_autovacuum FROM pg_stat_user_tables ORDER BY n_dead_tup DESC;

-- Cache hit ratio (aim > 99%)
SELECT sum(blks_hit) * 100.0 / nullif(sum(blks_hit + blks_read), 0) FROM pg_stat_database;

-- Replication lag
SELECT client_addr, state, replay_lag FROM pg_stat_replication;
```

For Prometheus, run `postgres_exporter` and alert on connections vs max, replication lag, slot WAL retained, transaction ID age, and long-running transactions.

## Interview quick answers

| Question | Short answer |
| --- | --- |
| How does MVCC work? | Writes create new row versions tagged with `xmin`/`xmax`; each transaction reads from a snapshot; VACUUM removes dead versions |
| Why does a table bloat? | Dead tuples outpace autovacuum, often due to long transactions; fix with tuned autovacuum, `pg_repack` |
| Why is my index not used? | Low selectivity, stale stats, function on column, type mismatch, or leading column of composite missing |
| `OFFSET` pagination is slow, why? | It still reads and discards skipped rows; use keyset pagination on an indexed column |
| Postgres vs MySQL? | Richer types (JSONB, arrays, ranges), extensions, SSI, transactional DDL; MySQL simpler replication, clustered PK |
| How do you avoid double booking? | Unique constraint, `SELECT ... FOR UPDATE`, Serializable, or an exclusion constraint on `tstzrange` |
| How do you scale writes? | Batch and tune WAL first, then partition, then shard (Citus or app-level by tenant) |
| How do you do zero-downtime migrations? | Add nullable columns, backfill in batches, `CREATE INDEX CONCURRENTLY`, `NOT VALID` then `VALIDATE` constraints, `lock_timeout` |
| What is WAL used for? | Durability, crash recovery, streaming replication, PITR, logical decoding / CDC |
| When pick Redis over Postgres? | Sub-ms hot reads, counters, rate limits, ephemeral data; keep Postgres as the source of truth |
