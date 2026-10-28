# Better raw forecasts did not improve default ranking

Allfivecontext-onlyfeatureestimators,120applicantindependentscalarchecks,10predictivemodelfits,10serializedmodelreplays andindependentfull70000OOFmetricchecks passed.

| Fixed pipeline | KS | AUC | Log loss |
|---|---:|---:|---:|
| Repaired V25 baseline | .5302185367 | .8470258500 | .2926003248 |
| Maturity-only candidate | .5306361042 | .8470560913 | .2925702718 |
| Conditional next-month account features | .5303064195 | .8468031384 | .2927434775 |

Do not promote the conditional-account recipe. Its smallKSgainversusoriginal is pairedwithanAUCloss and it trails the currentfeaturelead onboth. All model parameters, seeds, foldassignments, trainingbudgets,18/77featurewidths and75:25weights stayedfixed. Kumo90predictionswereunchanged. Onlythefivepredeclaredincome-relatedaliases changed.

The raw-observationaudit remains valid: correlatednoise reduced threeheldoutmonthMSE byabout8%/16% comparedwithiidnoise. The newresult shows this was insufficient justification for assuming improvedcreditriskfeatures. A conditionalnextmonthobservation includes short-lived accountstate. It is distinct from a borrower-level persistentfinancialtrait; the experiment doesnotidentifywhichlatentquantity drivesdefault. No causal claim or automatic follow-up promotion is inferred.

Eachfold's source-measurementparameters,featuretable andsourcehashes remainin fold*/. All5heads/5splines remainin models/fold*/. Feature verification isINDEPENDENT_FEATURE_AUDIT.json; fullmodel/OOFverification isINDEPENDENT_AUDIT.json andFINAL_STATUS.json. Uncertainty isconditionalonpredictions,notcorrectedforhistoricalsearch orretraining. Defaultlabels wereexcludedfromfeature-estimationandquerycovariatesdidnotfitsharedparameters. No confirmation orsubmission wasopened/generated.

Target.54 remainsunmet. Thebestfeature-onlypointresultremains.5306361042KS/.8470560913AUC withintervalsincludingnogain. Preserveallnegativeevidence andtheoriginaldeadline.
