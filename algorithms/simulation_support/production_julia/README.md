# Julia simulation backend

`simulate.jl` is the compiled implementation used by the public Python entry
point. It requires Julia 1.7 or newer and its standard library only. The Python
entry point normalizes edge lists with the same loader as the exact algorithm,
starts each independent replicate with an explicit seed, and writes a CSV plus
the parameter and uncertainty summary.

From the repository root:

```sh
julia --startup-file=no algorithms/simulation_support/production_julia/test_simulation.jl
python algorithms/simulate.py --edge-file examples/pa_n100_k4_seed42.edgelist --backend julia --rule DB --payoff average --benefit 4.5 --cost 1 --delta 0.01 --mutation-rate 0.0001 --steps 100000 --replicates 2 --seed 42 --output outputs/simulation.csv --trace-dir outputs/traces
```

For a paper-scale trajectory, specify the required `--steps`, `--replicates`,
network and benefit explicitly. Fig. 2 simulations used 10^12 updates per
replicate, 10 independent replicates, delta=0.01 and mu=0.0001. Generating a new
graph or using new trajectory seeds produces an independent reproduction, not
the identical archived measurements. The short command above checks operation;
it does not establish stationary convergence. Replicates run sequentially;
separate commands with distinct seeds and output paths can be scheduled in
parallel on a cluster. `--record-every` affects diagnostic trace output only;
every measured post-update state contributes to the cooperation mean.

## Model

- Average payoff: `b * cooperative_neighbours / degree - c * own_strategy`.
- Accumulated payoff: `b * cooperative_neighbours - c * degree * own_strategy`.
- Fitness: `exp(delta * payoff)`, evaluated with a stable exponential shift.
- DB: choose a focal node uniformly and a copying source among its neighbours
  with probabilities proportional to fitness.
- IM: choose the copying source among the focal node and all its neighbours
  with probabilities proportional to fitness.
- PC: choose a neighbour uniformly; copy it with probability
  `1 / (1 + exp(delta * (focal_payoff - neighbour_payoff)))`, otherwise copy self.
- After this source selection, with probability mu reset the focal strategy to
  uniform C/D. This also applies when PC would retain its strategy.
- Initial states are independent Bernoulli(0.5) by default. No burn-in is
  discarded by default; all holds and strategy changes are counted.

## Provenance and adaptations

The update kernels were adapted from the project's fixed-network production
implementation, identified by these source-relative names and SHA256 hashes:

| Source | SHA256 |
|---|---|
| `sigma/server_jobs/fixed_AB_RR_DB_PC_IM_20260907/simulate_fixed_graph.jl` | `9b2442ef73114bf04cdbcc76d73ffb30ff86e28bc3daa55da68f856d41cee6ec` |
| `sigma/server_jobs/fixed_AB_RR_DB_PC_IM_20260907/legacy_core.jl` | `48c6cb1e1cb75609d342c335fb7fdf53fc2ecef5e5c3c22f4cfb965300ae88d7` |

This distribution extracts DB, PC and IM and removes server orchestration,
unrelated birth-death code, network generation and fixed filenames. Cost,
mutation rate, initialization, burn-in and run length are parameters. Integer
neighbour cooperation counts replace incremental floating-point payoff updates
to prevent accumulated rounding drift; Int128 cooperation sums avoid Int64
overflow in very long trajectories. The transition probabilities are unchanged.
Julia uses an explicit Xoshiro RNG. The original production implementation used
the default RNG, with a different random-call order for PC, so identical seeds
do not imply identical trajectories. Python uses NumPy's default generator.

## Validation status

The Python reference passed independent small-graph transition-probability
checks for all six rule/payoff combinations, incremental-state checks,
reproducible-seed checks and hold-counting checks. Julia has a corresponding
standalone self-test above. On 15 September 2026, the extracted backend passed
both self-test groups (180,030 and 6,027 assertions) with Julia 1.13.0 on Windows.
The Python-to-Julia CLI also completed two 1,000-update replicates on the fixed
100-node example, including CSV/JSON output and a final partial trace block.
The official portable runtime archive was checked against its published SHA256.
No long production simulations were rerun while preparing this repository.
