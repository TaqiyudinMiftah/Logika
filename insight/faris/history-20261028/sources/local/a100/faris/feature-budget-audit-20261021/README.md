# Information overlap in the six V25 additions

No predictive model was fitted. These audits read only the explicit50000context applications in each of the five frozen folds; applicant default labels were not loaded.

Two of the six V25 additions overlap information already present in Kumo90:

| V25 correction input | Existing Kumo90 input | Verified relationship |
|---|---|---|
| aux_account_months | has_account | Exactly3times the existing flag in all five context tables |
| aux_ecommerce_observed_days | v2_ecom_observed_days | Identical after the actual SDM TableTensor conversion to torch.float32, with identical missing masks |

The raw shopping-span difference reaches about.00002984days because the inherited bank rounded the same measure to float32. It does not survive the model's actual numerical conversion. The account relationship is a supplied-dataset result; incomplete future histories need not obey it. Neither statement certifies future-data equivalence.

This does not contradict V25's benefit: its original12-field correction did not explicitly contain these inputs. New information is relative to the consuming representation. Adding six fields to that correction is not the same information expansion as adding six fields to the existing90-field Kumo table. Duplicate information can still change finite-model predictions; this audit does not prove why the earlier96-column trial failed to improve KS.

The four distinct coverage candidates relative to Kumo90 are:

* risk_conditional_negative_months: eligible due months with negative completed-month cash flow and at least one recorded allocation.
* risk_conditional_nonnegative_months: corresponding nonnegative-cash-flow months.
* aux_orders90: completed orders in the last90days. The existing30/60day order counts have a different window/status meaning.
* aux_allocation_count: installments with a recorded allocation timestamp. The existing total installment count is a different observation.

All six remain in the frozen correction; no input or model is changed by this audit. A recorded allocation is not proof of timely/full settlement. These raw features were independently reconstructed earlier in ../coverage-transfer-20261021.

The separate predefined-pair audit found five, not six, related legacy pairs with Pearson>=.98 in every context. Four are price-weighting variants, one is the1day/7day delinquency summary. The category-weighting/median pairs were below.98, so a six-slot replacement was not supported by this criterion. High correlation is not predictive equivalence: order/item/category weighting can retain useful differences.

A possible controlled hypothesis is to place the four distinct coverage fields into four highly correlated slots while retaining exactly90inputs, with a same-width negative control. Any learned slot selector must use only its own context and a frozen criterion. No global feature selection, experiment admission, default-label screen or performance claim has occurred here. The current missing-mask experiment must finish before another predictive recipe is considered.

Evidence: RESULT.json (seven predefined pairs, fivecontexts); SOURCE_REDUNDANCY.json (functional overlap and actual TableTensor precision); source/audit.py and source/source_redundancy.py. The latter two preserve their exact source hashes. This is a representation finding, not an improvement in KS/AUC.
