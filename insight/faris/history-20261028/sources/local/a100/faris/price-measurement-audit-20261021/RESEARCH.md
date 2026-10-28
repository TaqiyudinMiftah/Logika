# Price rounding and the observed lower edge

Before another classifier fit, inspect whether the existing additive log-price estimator has a material observation-model issue. Source inspection confirms it logs quantized unit prices, estimates item/order variance and an empirical-Bayes borrower effect, and reports the unit-price mass at5000. It does not include an explicit rounded/censored likelihood. This is a measurement hypothesis, not evidence that clipping caused a scoring gap.

Use only originalfold0's50000context customers for shared diagnostics. No default label, correction customer, validation customer or test customer enters these estimates. Verify every transaction predates its decision, quantities/unitprices are positive, item identifiers are unique, and unit amounts lie on the observed grid. Report exact minima and masses at5000andneighboringbins, bycategory/status, plus the number/fraction of customers affected and their own floor share. Cancelled baskets are observations of price preference, not actual spending; retain status distinctions.

An observed lower edge and a mass point are not proof of a particular generator, censoring rule or latent lognormal distribution. Do not replace values with arbitrary numbers, infer RNGs/labels, or admit a censor-correction model solely because a mass point exists. Check whether enough customers/categories are affected and whether the same price estimate is already repeated across independent orders. Retain allrawhashes and exactcounts.

Target.54KS remainsunmet. The negative income-measurement branch remainsclosed. This audit fits no creditmodel, changes no prediction, reads no confirmation, and produces no submission. The originaldeadline/admissionstop remainunchanged.
