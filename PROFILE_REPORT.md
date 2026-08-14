# Focused performance profile

## Scope and method

This is a small `cProfile` study of normal algorithm calls, with no algorithm
implementation changes. It uses cached, resolved benchmark fixtures:

- `HennWaescherUniform/1l-20-30-0.txt`: 5 orders, 54 pick positions;
- `MuterOencan/100_48_5.txt`: 14 orders, 49 pick positions.

The reported runtimes are medians of unprofiled cold calls (new algorithm
instance per call). `cProfile` was then run once per case to identify the call
paths. Local search used its normal route evaluator and a five-second limit;
both fixtures completed well below that limit. Nearest-neighbour routing used
the benchmark configuration with route and item-sequence output disabled.

## Runtime

| Algorithm | Henn median | Muter median | Notes |
| --- | ---: | ---: | --- |
| Order-number FIFO batching | 0.12 ms | 0.24 ms | 12 cold repetitions |
| Nearest-neighbour routing, all fixture picks | 0.39 ms | 0.30 ms | 8 cold repetitions |
| Local-search batching + Ratliff--Rosenthal routing | 3.96 ms | **70.38 ms** | 3 / 2 cold repetitions |
| Clark--Wright savings batching + nearest-neighbour routing | 2.23 ms | **20.27 ms** | 3 / 2 cold repetitions |
| LPT scheduling, fixture jobs | 0.009 ms | 0.014 ms | 30 cold repetitions |

The two batching algorithms that repeatedly score candidate batches dominate
this representative study. Exact/MIP was deliberately omitted: it is not
needed to identify the hot paths and would measure solver behaviour rather
than the shared Python implementation paths.

## Expensive call paths (Muter fixture)

`cProfile` time is useful for attribution but is inflated relative to the
unprofiled medians; use the percentages and ordering, not its absolute time.

### Local search + Ratliff--Rosenthal

One profiled solve made 206 `_batch_cost_from_orders` calls, of which 54 were
route-cache misses and invoked Ratliff--Rosenthal. Its 182 ms profiled total
was approximately:

| Attribution | Evidence | Rough share |
| --- | --- | ---: |
| Algorithm/search work | Rebuilding DP state spaces: 113 ms cumulative; Bellman--Ford shortest paths: 40 ms cumulative | ~80--85% |
| Route/result construction | `_construct_picker_tour` 8.7 ms, item-sequence extraction 3.4 ms, annotated route 5.9 ms | ~10% |
| Capacity computation | `orders_fit` / consumption: 3.6 ms cumulative, 143 checks | ~2% |
| Candidate/cache-key/object preparation | `_batch_cost_from_orders` self time: 0.5 ms | <1% measured directly |
| Repeated pick traversal | `_get_aisle_orders`: 1,620 calls, 1.6 ms self; also contributes inside DP helpers | small by itself |

The cache avoided roughly 74% of route evaluations (152 hits / 206 cost
requests), but the 54 misses still determine runtime. Capacity checks occur
frequently but are not the limiting factor here.

### Clark--Wright + nearest-neighbour

One profiled solve made 504 route-cost requests; 182 were cache misses, so the
route cache avoided roughly 64% of potential routing calls. Its 39 ms profiled
total was approximately:

| Attribution | Evidence | Rough share |
| --- | --- | ---: |
| Algorithm/search work: repeated NN route evaluations | `routing.solve`: 32.7 ms cumulative across 182 routes | ~84% |
| NN candidate scanning and order-list removal | nearest-node selection 10.5 ms self; walk/removal path 17.7 ms cumulative | dominant within routing |
| Capacity computation | `orders_fit`: 2.9 ms cumulative, 169 checks | ~7% |
| Candidate construction, order-id cache keys, flattening | route-cost helper: 1.2 ms self | ~3% |
| Distance access | `_get_distance`: 1.0 ms self, 3,295 lookups | ~3% |

### Standalone nearest-neighbour routing

For the full 49-pick Muter list, `_walk_to_target` (including traversal to
remove picked positions) accounted for 0.39 ms cumulative and nearest-candidate
index-list construction/scanning 0.20 ms. Raw numeric distance lookup was only
0.014 ms. Router construction, including obtaining `distance_matrix.values`
and building the node index, was about 0.034 ms in the profiled full lifecycle
(~5%); the batching algorithms construct that router once and reuse it.

There were no pandas scalar distance accesses in these hot calls: routing reads
the NumPy array cached from the DataFrame. Storage mappings and item assignment
were intentionally completed before the algorithms, as in the benchmark
fixtures, and did not appear in the selected call profiles.

## Optimization decision

1. **Highest value: compact/reusable representation for repeated
   Ratliff--Rosenthal route evaluations.** Local search rebuilds NetworkX state
   graphs and reruns Bellman--Ford for each cache miss. The evidence supports an
   internal compact search representation or other reuse that preserves exact
   routing semantics.
2. **Second: make repeated heuristic route scoring cheaper.** In savings
   batching, nearest-neighbour candidate scanning plus current-order traversal
   dominates the 182 route misses. Focus on the scoring path and its temporary
   pick/index lists before general batching-object redesign.
3. **Third, modest: precompute per-order capacity consumption.** It is a clear
   but secondary cost (about 7% in Clark--Wright and 2% in local search), with a
   much larger relative effect on cheap FIFO batching. Precomputation is worth
   considering only after route-evaluation work.

The evidence **does justify** compact internal search representations for the
repeated Ratliff--Rosenthal and nearest-neighbour scoring paths, and **modestly
supports** capacity precomputation. It **does not justify** a broad replacement
of shared distance/layout access: pandas scalar lookup is absent, raw NumPy
distance access is only a few percent, and node-index setup is amortized. It
also does not support a broad WarehouseOrder/PickPosition reconstruction rewrite
at this point; flattening/allocation is visible but small relative to routing.

Reproduce with `uv run python tools/profile_representative.py --fixture muter
--expensive-repeats 2` (or `--fixture henn`).

## RR direct-DP follow-up

The Ratliff--Rosenthal NetworkX state graph and Bellman--Ford scoring kernel
were subsequently replaced by direct seven-state layered propagation. On the
same Muter fixture and in the same session (five local-search repetitions):

| Measurement | NetworkX baseline | Direct DP | Ratio |
| --- | ---: | ---: | ---: |
| Standalone RR, 49 picks | 1.149 ms | 0.194 ms | 5.9x faster |
| Local search + RR | 59.053 ms | 9.518 ms | 6.2x faster |

Post-change profiling contains only aisle-layer propagation, cross-aisle-layer
propagation, aisle preparation, and predecessor backtracking in the RR scoring
path. NetworkX node/edge construction and Bellman--Ford calls are absent.

## Final stateless routing-core results

The completed rewrite separates scalar scoring, semantic result construction,
and physical path materialization. Routers retain configuration and the normal
`Algorithm` timing fields only. Constructive algorithms use local node/aisle
collections; standard RR uses the direct seven-state DP; exact TSP and
scattered RR use algorithm-specific solve-local model data. Profitable SPRP is
the intentional exception: its reusable model and graph live in the context
returned by `prepare()`, never on the router.

The following medians use a reused router and all fixture picks. Columns are
`score()` / `solve()` without output / `solve()` with tour and item sequence.

| Router | Henn (ms) | Muter (ms) |
| --- | ---: | ---: |
| S-shape | 0.033 / 0.033 / 0.327 | 0.029 / 0.031 / 0.160 |
| Return | 0.031 / 0.034 / 0.453 | 0.028 / 0.030 / 0.186 |
| Midpoint | 0.053 / 0.055 / 0.414 | 0.049 / 0.055 / 0.212 |
| Largest gap | 0.047 / 0.048 / 0.371 | 0.044 / 0.048 / 0.207 |
| Nearest neighbour | 0.133 / 0.148 / 0.431 | 0.118 / 0.132 / 0.252 |
| U-shape | 0.028 / 0.029 / 0.320 | 0.029 / 0.027 / 0.168 |
| Pick-list order | 0.036 / 0.038 / 0.806 | 0.020 / 0.021 / 0.295 |
| Ratliff--Rosenthal | 0.149 / 0.148 / 0.600 | 0.150 / 0.152 / 0.718 |

Minimal semantic construction adds almost nothing. Physical predecessor-path
or Euler-tour expansion is now paid only when requested. The final FIFO-batch
routing stage with nearest neighbour took 0.243 ms on Henn and 0.301 ms on
Muter, so the cheaper search API did not move work into or degrade normal final
route production.

### End-to-end candidate search

| Pipeline | Henn baseline / final | Muter baseline / final |
| --- | ---: | ---: |
| Local search + RR | 3.96 / 0.65 ms | 70.38 / 8.59 ms |
| Clark--Wright + NN | 2.23 / 0.73 ms | 20.27 / 8.12 ms |

On the Muter local-search profile there were 206 route requests, 152 cache hits,
and 54 direct-RR score calls. Scalar routing consumed 18.45 ms of the 21.54 ms
profiled call (~86%): `_solve_direct_dp` was 16.01 ms cumulative, cross-aisle
propagation 8.66 ms, aisle propagation 7.00 ms, and aisle preparation 2.38 ms.
Capacity checking was 2.21 ms cumulative. These cumulative figures overlap.

Clark--Wright made 504 requests, with 322 hits and 182 NN score calls. Scalar
routing consumed 9.93 ms of the 13.18 ms profiled call (~75%); the compact
`_nearest_kernel` accounted for 9.80 ms. Capacity checking was 1.96 ms, while
the remaining distance accessor calls were only 0.086 ms. Candidate evaluation
constructs no `Route`, `RoutingSolution`, annotated path, or predecessor path;
this is also guarded by tests that replace `Route` with a failing sentinel.

NetworkX state construction and Bellman--Ford are absent from the standard RR
scoring profile. NetworkX remains only in scattered/profitable formulations and
requested Euler-tour reconstruction, where it represents algorithm-specific
model/output work rather than the hot scoring kernel.

Reproduce the concise matrix with:

```text
uv run python tools/profile_representative.py --fixture henn --summary-only
uv run python tools/profile_representative.py --fixture muter --expensive-repeats 5 --summary-only
```
