# Band-only ablation: not promoted

Both five-fold conditions finished before scoring. Real-time bands reached **KS 0.5295255960 / AUC 0.8471189857 / log loss 0.2926064402**. Their matched shuffled-time control reached **0.5271689683 / 0.8464569906 / 0.2930109385**.

Compared with the maturity candidate (0.5306361042 / 0.8470560913), removing the four selected-extrema fields improves AUC but lowers pooled KS. It wins three KS folds and two AUC folds, with the AUC increase concentrated in fold five. KS fold deltas: [-.00152725, +.00037456, -.00174951, +.00161927, +.00341341]. AUC fold deltas: [-.00008983, +.00001643, -.00014848, -.00005611, +.00060212].

The opposite metric changes for full versus reduced spectral features do not establish a joint improvement. Neither recipe is promoted. The real-time ordering beats the single shuffled control at point estimates, but comparison with a deliberately destroyed signal does not establish superiority to the stronger baseline. These are adaptively chosen features evaluated on historically reused labels.

Close the periodicity branch. No further frequency, window or feature-subset search follows. Preserve all 20 models and source/feature hashes. No confirmation or submission is opened/generated. Target 0.54 remains unmet; the maturity candidate remains the provisional joint lead, with uncertainty including no improvement.

Independent verification passed: all 20 serialized models replayed exactly, original resolved parameters/objectives/source hashes checked, complete 70,000-row predictions reconstructed for each condition, and empirical-CDF KS / rank AUC matched. See FINAL_STATUS.json and INDEPENDENT_AUDIT.json.
