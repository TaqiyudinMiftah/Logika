# Population context follow-ups

KumoLarge100% frozen; identical outer folds and original90 inputs. These are frozen contextual experts, not a LoCalPFN reproduction or fine-tuning. Global anchor85% and localweight15% matchV178. Unsupported/missing group retains global scores. Every source index is an audited subset of original50k context, excluding6k correction and14kvalidation.

## V222: Employment-conditioned frozen Large population context

The three employment populations differ in income regularity and affordability mechanisms. Observed labeled sizes33473/20987/15540 and targetrates.132405/.137943/.140283 motivate conditional relationships, not a causal employment claim. V178 grouped only history coverage; test a different population axis using unchanged90inputs.

Formula: `Group only original50k context by observedemployment; cap24k pergroup, keepnatural target proportions,8 frozenestimators; rawKumo=.85globalV168+.15matchinglocal, refit original18-input6k head and75:25spline77fusion`

## V223: Underwriting-regime frozen Large population context

APR is discretized: train/test quantiles include.16,.20,.24,.32 and median.20; no missing APR. Test whether local underwriting regimes help interpret bureau/burden signals. APR is policy-generated and could drift; no claim that it independently measures risk, and grouping never uses outer target.

Formula: `For each fold, compute medianAPR only within50k context; low/equal vs high groups, cap24k and8 frozenestimators; .85global+.15matchinglocal before original18-input correction and75:25spline77fusion`

RAM guard requires2.8GiB effective capacity, ownRSSceiling3GiB, GPUfree24GiB. Stream pinned-weight checksum instead of reading850MB atonce. Original histories/global latent-reference uncertainty remains inherited. No test or validation labels choose population references.
