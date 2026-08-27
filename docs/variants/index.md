# Choosing a variant

| If | Use | Why |
|---|---|---|
| Your cases are pass/fail or integer counts | [`lexicase_selection`](lexicase.md) | Exact ties are meaningful, so plain filtering works |
| Your cases are continuous errors | [`epsilon_lexicase_selection`](epsilon.md) | Exact ties never happen with floats, so plain lexicase degenerates to "best on the first case" |
| Evaluation is the bottleneck | [`downsample_lexicase_selection`](downsample.md) | Uses a random subset of cases per event, so you can run more generations for the same evaluation budget |
| Evaluation is the bottleneck and your cases are redundant | [`informed_downsample_lexicase_selection`](informed-downsample.md) | Picks a subset whose cases disagree with each other, instead of a uniform sample |
| You want the subsampling saving but every case used somewhere | [`cohort_lexicase_selection`](cohort.md) | Splits the population and the cases into paired cohorts |
| Lexicase is too strict and kills your diversity | [`batch_lexicase_selection`](batch.md) | Filters on the mean over a batch of cases, so batch size tunes selection pressure |
| Selection is the bottleneck, not evaluation | [`dalex_selection`](dalex.md) | One softmax and one matrix multiply |
| You want the selection distribution itself | [`plexicase_probabilities`](plexicase.md) | The approximate per-individual selection probability, without drawing anything |
| Some cases matter more than others | [`case_weights=`](case-weights.md) | Weighted case ordering instead of a uniform shuffle |

If you are not sure: `epsilon_lexicase_selection` for regression and anything
continuous, `lexicase_selection` for program synthesis and anything discrete. Add
`downsample_size` once evaluation cost starts to hurt.

## Selection pressure, roughly ordered

From loosest to strictest, on a fixed population:

1. `batch_lexicase_selection` with a large batch size, which becomes elitist
   selection on mean fitness
2. `dalex_selection` with low particularity pressure, which approaches the same
   thing
3. `epsilon_lexicase_selection` in `"static"` mode
4. `epsilon_lexicase_selection` in `"semi-dynamic"` and `"dynamic"` modes
5. `lexicase_selection`, `plexicase_selection`, and `dalex_selection` with high
   particularity pressure, which all target the same distribution

`downsample_lexicase_selection` and `cohort_lexicase_selection` are not on this
axis. They trade evaluation cost for noise in the selection, which usually shows
up as more diversity rather than less pressure.

`examples/08_variant_tour.py` prints the selection share every method gives every
individual on one small population, which is the fastest way to build intuition
for these differences.
