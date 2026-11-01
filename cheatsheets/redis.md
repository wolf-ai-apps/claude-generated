# Redis Cheat Sheet

As of Nov 1, 2026. Editable version: [Redis Cheat Sheet doc](https://claude.ai/artifact/X22baqCEvtNpz873ghyLTF)

## Mental model

Redis is an in-memory data structure server: one thread runs every command, so each command is atomic and any slow command blocks all clients.

| Idea | What it means | Consequence |
| --- | --- | --- |
| In-memory | Dataset lives in RAM; disk is optional backup | Sub-millisecond ops; dataset must fit in memory |
| Single-threaded execution | Commands run one at a time (I/O threads exist since 6.0, execution stays single) | No races inside a command; O(N) commands hurt everyone |
| Data structures | Server-side lists, sets, hashes, sorted sets, streams | Mutate in place instead of read-modify-write in the app |
| RESP protocol | Simple text protocol over TCP, port 6379 | Easy to debug with `redis-cli`, easy to reimplement |

```bash
docker run -d --name redis -p 6379:6379 redis:7
docker exec -it redis redis-cli
```

## Keys and TTL

Name keys `object:id:field` (e.g. `user:42:sessions`) and give every cache key a TTL.

| Command | Does | Note |
| --- | --- | --- |
| `EXISTS k` | 1 if key exists | Accepts many keys |
| `DEL k` / `UNLINK k` | Delete key | `UNLINK` frees memory in background; prefer for big keys |
| `TYPE k` | Returns string, hash, list... | |
| `EXPIRE k 60` / `PEXPIRE k 60000` | Set TTL in s / ms | `SET` without `KEEPTTL` clears TTL |
| `TTL k` | Seconds left | -1 = no TTL, -2 = key missing |
| `PERSIST k` | Remove TTL | |
| `RENAME a b` | Rename | Overwrites `b` |
| `SCAN 0 MATCH user:* COUNT 100` | Iterate keys with a cursor | Use instead of `KEYS *` in production |
| `OBJECT ENCODING k` | Internal encoding | listpack, skiplist, hashtable, intset... |

Expiry is lazy (checked on access) plus active (background sampling of keys with TTLs), so expired keys can briefly use memory.

## Data types

Pick the type by the question you need answered: lookup by key (string/hash), order by arrival (list/stream), membership (set), or rank by score (sorted set).

| Type | Core commands | Complexity | Use for |
| --- | --- | --- | --- |
| String (up to 512 MB) | `SET k v EX 60 NX`, `GET`, `MGET`, `INCR`, `INCRBY`, `GETDEL`, `APPEND` | O(1) | Cache values, counters, locks, JSON blobs |
| Hash | `HSET h f v`, `HGET`, `HGETALL`, `HINCRBY`, `HDEL`, `HEXPIRE` (7.4+) | O(1) per field; `HGETALL` O(N) | Objects, sessions, per-user settings |
| List | `LPUSH`, `RPUSH`, `LPOP`, `RPOP`, `BLPOP k 5`, `LRANGE k 0 -1`, `LTRIM`, `LMOVE` | Push/pop O(1); index O(N) | Queues, recent-N feeds |
| Set | `SADD`, `SREM`, `SISMEMBER`, `SMEMBERS`, `SCARD`, `SINTER`, `SUNION`, `SRANDMEMBER` | O(1) add/check | Tags, unique visitors, friends in common |
| Sorted set | `ZADD z 10 m`, `ZINCRBY`, `ZRANGE z 0 9 REV WITHSCORES`, `ZRANK`, `ZRANGEBYSCORE`, `ZREMRANGEBYSCORE`, `ZPOPMIN` | O(log N) | Leaderboards, sliding-window limits, delayed jobs |
| Stream | `XADD s * f v`, `XRANGE`, `XGROUP CREATE`, `XREADGROUP`, `XACK`, `XPENDING`, `XAUTOCLAIM` | O(1) append | Event log, durable queue with consumer groups |
| Bitmap | `SETBIT`, `GETBIT`, `BITCOUNT`, `BITOP` | O(1) bit ops | Daily active flags, feature flags |
| HyperLogLog | `PFADD`, `PFCOUNT`, `PFMERGE` | O(1), 12 KB, ~0.81% error | Approximate unique counts |
| Geo | `GEOADD`, `GEOSEARCH ... BYRADIUS 5 km` | O(log N) | Nearby stores and drivers |
| Pub/Sub | `PUBLISH`, `SUBSCRIBE`, `PSUBSCRIBE` | O(subscribers) | Fire-and-forget fan-out; no persistence |

Small hashes, lists, sets and sorted sets use compact listpack/intset encodings and switch to hashtable/skiplist once they pass size thresholds (`*-max-listpack-entries`).

## Atomicity: pipeline vs transaction vs Lua

Use a pipeline for speed, `MULTI/EXEC` for all-or-nothing batches, and Lua (or Functions) when a write depends on a read.

| Tool | Atomic | Can branch on values read | Purpose |
| --- | --- | --- | --- |
| Pipeline | No | No | Cut round trips: send N commands, read N replies |
| `MULTI` ... `EXEC` | Yes, queued commands run back-to-back | No | Group writes; no rollback if one command errors |
| `WATCH k` + `MULTI/EXEC` | Optimistic (CAS) | Yes, in the client; retry if `EXEC` returns nil | Check-and-set without Lua |
| `EVAL` Lua / `FUNCTION` | Yes, whole script | Yes | Read-modify-write on the server, e.g. a safe lock release |

```lua
-- Release a lock only if we still own it
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
```

In Redis Cluster, every key in one transaction or script must live in the same hash slot; use hash tags like `{user:42}:cart`.

## Common patterns

Most Redis designs are one of these patterns.

| Pattern | Recipe | Gotcha |
| --- | --- | --- |
| Cache-aside | `GET`; on miss read DB, then `SET k v EX ttl`; on write update DB, then `DEL k` | Stampede on hot key expiry: add TTL jitter, a lock, or early refresh |
| Fixed-window rate limit | `INCR rl:{user}:{minute}`; if result is 1, `EXPIRE 60`; reject if > limit | Bursts of 2x at window edges |
| Sliding-window rate limit | `ZREMRANGEBYSCORE k 0 now-window`, `ZADD k now id`, `ZCARD k` in Lua | Memory grows with requests per window |
| Distributed lock | `SET lock:x <uuid> NX PX 30000`; release with the Lua script above | Not safe if a GC pause outlives the TTL; use fencing tokens for correctness |
| Simple queue | `LPUSH jobs j` / `BRPOP jobs 0` | Job lost if worker dies; use `LMOVE` to a processing list or Streams |
| Reliable queue | `XADD` + `XREADGROUP` + `XACK`; `XAUTOCLAIM` reclaims stuck jobs | Trim with `XADD ... MAXLEN ~ 100000` |
| Leaderboard | `ZINCRBY lb 10 user`; `ZRANGE lb 0 9 REV WITHSCORES`; `ZREVRANK lb user` | Ties sort by member name |
| Session store | `HSET sess:<id> ...` + `EXPIRE` on each request | Set `maxmemory-policy` so sessions are not evicted early |
| Delayed jobs | `ZADD delayed <run_at> job`; poll `ZRANGEBYSCORE delayed 0 now` | Pop atomically with Lua or `ZPOPMIN` checks |

```python
import redis
r = redis.Redis(decode_responses=True)

def get_user(uid):
    key = f"user:{uid}"
    if (v := r.get(key)) is not None:
        return v
    v = db_load_user(uid)          # your DB call
    r.set(key, v, ex=300)
    return v
```

## Persistence, eviction and scaling

Redis trades durability for speed by default: replication is asynchronous, so a failover can lose acknowledged writes.

**Persistence**

| Mode | How | Data loss on crash | Cost |
| --- | --- | --- | --- |
| RDB | `fork()` + copy-on-write snapshot (`save 3600 1`, `BGSAVE`) | Everything since last snapshot | Fork latency; up to 2x memory under heavy writes |
| AOF `appendfsync everysec` | Log every write; rewrite compacts it | About 1 second | Default sensible choice |
| AOF `always` | fsync each write | Near zero | Much slower |
| RDB + AOF | Both (AOF base is RDB since 7.0) | About 1 second | Recommended for durable setups |

**Eviction** (`maxmemory 2gb`, `maxmemory-policy`)

| Policy | Evicts | Use when |
| --- | --- | --- |
| `noeviction` (default) | Nothing; writes error | Redis is a primary store |
| `allkeys-lru` / `allkeys-lfu` | Any key, least recently / frequently used | Pure cache |
| `volatile-lru` / `volatile-ttl` | Only keys with a TTL | Mixed cache + persistent keys |

**Scaling and HA**

| Setup | What it gives | Key facts |
| --- | --- | --- |
| Replication | Read replicas, copies of data | `REPLICAOF host port`; async; `WAIT n ms` for stronger acks |
| Sentinel | Automatic failover for one primary | Run 3+ sentinels; clients ask Sentinel for the primary |
| Cluster | Sharding + failover | 16384 hash slots, `CRC16(key) % 16384`; clients follow `MOVED`/`ASK`; multi-key ops need one slot |

CAP view: Redis Cluster favors availability; during a partition a minority primary can accept writes that are later lost.

## Performance gotchas and observability

Most Redis incidents come from O(N) commands on big keys, memory pressure, or fork latency.

| Avoid | Why | Do instead |
| --- | --- | --- |
| `KEYS *` | O(N) over the whole keyspace, blocks server | `SCAN` with a cursor |
| `HGETALL` / `SMEMBERS` / `LRANGE 0 -1` on huge keys | Blocks and floods network | `HSCAN`, `SSCAN`, paginate |
| `DEL` on a key with millions of elements | Synchronous free | `UNLINK` or `lazyfree-lazy-*` settings |
| Big keys (> 1 MB or > 10k elements) | Uneven cluster slots, slow migrations | Split into buckets (`key:{n}`) |
| One round trip per command | Network-bound at ~10k ops/s | Pipeline or `MGET`/`MSET` |
| Hot key | One shard saturates | Client-side caching, replicas, key splitting |
| Swap enabled | Latency spikes | `maxmemory` below RAM; disable swap |

| Command | Shows |
| --- | --- |
| `INFO memory` / `INFO stats` / `INFO replication` | Memory, hit rate (`keyspace_hits` vs `keyspace_misses`), evictions, lag |
| `SLOWLOG GET 10` | Slowest recent commands |
| `LATENCY DOCTOR` | Latency spike analysis |
| `MEMORY USAGE k` / `redis-cli --bigkeys` / `--memkeys` | Size of keys |
| `CLIENT LIST` | Connected clients and buffers |
| `MONITOR` | Every command live; debug only, heavy |
| `redis-benchmark -q -n 100000` | Throughput baseline |

For Prometheus, run `redis_exporter` and alert on memory used vs `maxmemory`, evicted keys, replication lag, and blocked clients.

## Interview quick answers

| Question | Short answer |
| --- | --- |
| Why is single-threaded Redis fast? | RAM access, no lock contention, epoll event loop, efficient encodings; the bottleneck is usually network, not CPU |
| Redis vs Memcached? | Redis has data structures, persistence, replication, Lua, streams; Memcached is multi-threaded, simple string cache |
| Is Redis a database? | Yes for some uses (AOF + replicas), but async replication can lose writes; keep the source of truth elsewhere for money or orders |
| How do you keep cache and DB consistent? | Write DB, then delete cache key; short TTL as safety net; for strict needs use CDC (Debezium) to invalidate |
| Cache penetration / breakdown / avalanche? | Missing keys: cache nulls or a Bloom filter. Hot key expiry: mutex or early refresh. Mass expiry: TTL jitter |
| Design a rate limiter | Token bucket or sliding window in Lua, key per user + route, return remaining quota in headers |
| Design a leaderboard | Sorted set per board/season; `ZINCRBY`, `ZREVRANGE`, `ZREVRANK`; shard by board for scale |
| Is Redlock safe? | Fine for efficiency locks; for correctness use fencing tokens or a consensus store (etcd, ZooKeeper) |
| How does Cluster pick a node? | `CRC16(key) mod 16384` slot, slot mapped to a primary; clients cache the slot map |
| What happens when memory is full? | Depends on `maxmemory-policy`: error on writes (`noeviction`) or evict by LRU/LFU/TTL |
