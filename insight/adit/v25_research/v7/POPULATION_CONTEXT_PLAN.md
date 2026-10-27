# Population context follow-ups

KumoLarge100% frozen; identical outer folds and original90 inputs. These are frozen contextual experts, not a LoCalPFN reproduction or fine-tuning. Global anchor85% and localweight15% matchV178. Unsupported/missing group retains global scores. Every source index is an audited subset of original50k context, excluding6k correction and14kvalidation.

## V222: Employment-conditioned frozen Large population context

The three employment populations differ in income regularity and affordability mechanisms. Observed labeled sizes33473/20987/15540 and targetrates.132405/.137943/.140283 motivate conditional relationships, not a causal employment claim. V178 grouped only history coverage; test a different population axis using unchanged90inputs.

Formula: `Group only original50k context by observedemployment; cap24k pergroup, keepnatural target proportions,8 frozenestimators; rawKumo=.85globalV168+.15matchinglocal, refit original18-input6k head and75:25spline77fusion`

## V223: Underwriting-regime frozen Large population context

APR is discretized: train/test quantiles include.16,.20,.24,.32 and median.20; no missing APR. Test whether local underwriting regimes help interpret bureau/burden signals. APR is policy-generated and could drift; no claim that it independently measures risk, and grouping never uses outer target.

Formula: `For each fold, compute medianAPR only within50k context; low/equal vs high groups, cap24k and8 frozenestimators; .85global+.15matchinglocal before original18-input correction and75:25spline77fusion`

RAM guard requires2.8GiB effective capacity, ownRSSceiling3GiB, GPUfree24GiB. Stream pinned-weight checksum instead of reading850MB atonce. Original histories/global latent-reference uncertainty remains inherited. No test or validation labels choose population references.

## Controlled population-family ablation

### V225: Coverage and underwriting population contexts

History availability and underwriting regimes represent different mechanisms. Complete the paired population ablation with matched total15%local weight, unchanged data inputs, no optimized outer weights, and original6k correction.

Formula: `rawKumo=.5rawV178+.5rawV223=.85global+.075coverage-local+.075APR-local; refit18-input correction and75:25spline77fusion`

### V226: Occupation and underwriting population contexts

Test complementarity of employment-conditioned affordability and policy-conditioned context without the coverage expert. This completes a controlled context-family ablation; shared context examples are not independent evidence.

Formula: `rawKumo=.5rawV222+.5rawV223=.85global+.075employment-local+.075APR-local; refit18-input correction and75:25spline77fusion`

### V227: Three population views with a fixed global anchor

Compare all three population views to each matched pair under the same total15%local budget. No additional correction features or outer-fitted weights; source overlap and repeated selection remain caveats.

Formula: `rawKumo=mean(rawV178,rawV222,rawV223)=.85global+.05each coverage/employment/APR local; refit18-input correction and75:25spline77fusion`


## Five additional cached-score experiments V228-V232

No new backbone training or weight download. Controlled global-anchor ablations and one changed fusion stage, with no outer-fitted weights. Every run has its own submission, matched uncertainty, errors and overfitting report.

### V228: Conservative global/coverage/employment raw fusion

V222 employment context KS.529872 and AUC.846984 did not improveV25; test stronger global anchoring for the same coverage/employment views instead of trusting local retrieval at15%.

Formula: `raw=mean(globalV168,rawV178,rawV222)=.90global+.05coverage+.05employment; refit original18-input6k correction, L2.003/shrink.5; final75:25spline77; mean5foldtest`

### V229: Conservative global/coverage/underwriting raw fusion

Matched toV225 with total local influence reduced15% to10%; APR is policy-generated, so a stronger global anchor is a prespecified stability control, not an assumed gain.

Formula: `raw=mean(globalV168,rawV178,rawV223)=.90global+.05coverage+.05APR; refit original18-input6k correction, L2.003/shrink.5; final75:25spline77; mean5foldtest`

### V230: Conservative global/employment/underwriting raw fusion

Matched toV226 at a lower local budget; inspect whether conditional affordability/pricing information survives stronger shrinkage after employment-only context was negative.

Formula: `raw=mean(globalV168,rawV222,rawV223)=.90global+.05employment+.05APR; refit original18-input6k correction, L2.003/shrink.5; final75:25spline77; mean5foldtest`

### V231: Global anchor vote with all three population views

Complete the all-view ablation againstV227 by giving the global expert its own equal vote. Shared context borrowers are not independent evidence; fixed weights avoid an outer-validation weight search.

Formula: `raw=mean(globalV168,rawV178,rawV222,rawV223)=.8875global+.0375each local view; refit original18-input6k correction, L2.003/shrink.5; final75:25spline77; mean5foldtest`

### V232: Frozen Small/Large raw fusion before correction

V207 Small alone was weaker, whileV209 final-score blending gave a tiny numerical gain. Test combination before the18-input correction; changing fusion stage can expose complementary errors, but equal weights are not claimed optimal.

Formula: `raw=.50globalLargeV168+.50globalSmallV207; refit original18-input6k correction, L2.003/shrink.5; final75:25spline77; mean5foldtest`

