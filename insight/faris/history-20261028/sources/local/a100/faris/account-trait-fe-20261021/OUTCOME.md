# Persistent-trait account feature: completed negative evidence

Allfivefeaturetransforms reused the exact retainedcontext-onlymeasurementparameters. Independent covariance-form/precision-form posterior identities and120sampledscalarreconstructions passed. Thefixedcreditcomparison completed allfivefolds; all10serializedhead/splinemodelreplays and independent70000rowKS/AUCchecks passed inFINAL_STATUS.json andINDEPENDENT_AUDIT.json.

| Fixed pipeline | KS | AUC | Log loss |
|---|---:|---:|---:|
| Repaired V25 baseline | .5302185367 | .8470258500 | .2926003248 |
| Maturity-only feature lead | .5306361042 | .8470560913 | .2925702718 |
| Conditional monthly observation | .5303064195 | .8468031384 | .2927434775 |
| Persistent borrower randomintercept | .5296909044 | .8470043340 | .2926132906 |

Do not promote the persistent-trait recipe. It recovered someAUCrelative to the next-observationversion but remainsbelow theoriginalreference andcurrentlead. Themain architecture,hyperparameters,seeds,folds,trainingbudget,18/77inputwidths and75:25weights wereunchanged. Onlythefiveexplicitincomealiases changed. Learnedcovarianceswerereused,notrefitted; noquerysharedstatistics ordefaultlabelswererequiredforfeatureconstruction.

This closes the two testedaccountmeasurementestimands under this workingmodel. RawforecastMSEimprovement didnottranslateinto a defaultscoregain; changing the estimand from nextmonthobservation to a persistenttrait didnotresolve that. No further income-estimationvariant is admitted solely to extend this branch. It doesnotproveallaccountfeatureengineeringisuseless.

Everyfeaturetable,measurementparameterreceipt,head,spline,preprocessor,OOFprediction andconditionalinterval remainsretained. Historicaldefaultlabelreuse andadaptivehypothesischoice remainlimitations. No terminalconfirmation orcompetition submission wasopened/generated. Target.54 remainsunmet; thematurity-only.530636/.847056pointlead remainsuncertain.
