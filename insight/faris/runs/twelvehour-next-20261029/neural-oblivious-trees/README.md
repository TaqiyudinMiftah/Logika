# Sparse oblivious tree CANN,29October2026

192designs/1920plannedfits, not a completion claim. Sixexisting featureviews;16/32trees;depth3/5; fixedGLM or intercept offset; learned or frozen sparsemax featureselectors; sparsemoid or sigmoid routing. Allcomparisons use matched two seeds/fivefolds, same42500inner encoder/data initialization,7500stopping, fresh50000context refit. No new featurebank columns.

This is a one-layer NODE-inspired adaptation, not exact NODE: zero leafresponses, innerquantileinitialization and fullrefit reuse, AdamW, fixedGLM and KSstopping differ from officialbenchmark. Primarysources are Popov/Morozov/Babenko2019 (arxiv1909.06312v2), official Qwicen/node, and CANNactuarial reference doi10.1017/asb.2018.42.

Each fullsizepilot gates strictnumerical/boundary checks and conservativefinish before remaining9fits. Everyattempt retained; no automatic retry. Parameters/source/seeds/recipes and private compactpredictions saved, no fittedweights/KVcaches. CPUowners2001/2003/119/42 slots4/4/6/6, two threads each. All inherited absoluteclock limits unchanged: admission17:48:48UTC,models18:33:48UTC,all19:18:48UTC.

All70000rows adaptively reused development, no untouched audit. Prioroutcomes exposed; conditional intervals are not independentconfirmation. Identical fourcompositions and deterministic terminalunion. No refit or Kaggle action.
