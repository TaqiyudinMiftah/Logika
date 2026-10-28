# Real-time periodicity versus matched shuffled-time features

The raw180day11fieldfeaturebuilder reproduced all100000teammate rows with maximumerror8.33e-17. HistoricalV80heads/OOFwerereplayed beforethematchedexperiment. All20newhead/splinefits, sourcehashes, objectives, modelreplays and independent70000rowKS/AUCchecks passed.

| Fixed pipeline | KS | AUC | Log loss |
|---|---:|---:|---:|
| Repaired V25 baseline | .5302185367 | .8470258500 | .2926003248 |
| Maturity-only candidate | .5306361042 | .8470560913 | .2925702718 |
| Trueperiodicity plus maturity | .5309665315 | .8469295983 | .2927608823 |
| Shuffledtimecontrol plus maturity | .5267447011 | .8463162307 | .2931377503 |

Do not promote trueperiodicity: it improvespooledKS butlowersAUC/logloss andwins only2of5KSfolds versusmaturity-only. FoldKSdeltas are[-.00354958,-.00213429,-.00020482,+.00033333,+.00166803]. A higherpooledKS mustnot hide inconsistentfoldbehavior.

Trueversusshuffled conditional simultaneousintervals: deltaKS[-.00004291,.00733561], deltaAUC[.00010701,.00117617]. Thus AUCsupports anassociationwithretainedtimeordering underthissinglepredeclaredcontrol, but theadjustedKSintervalstillincludeszero. This isnotcausalevidence ofpayrollcycles, andthelargecontrolgapdoesnotestablishimprovementoverbaseline. Intervalsversusbaseline/maturityincludezero. Alllabels havehistoricalexposure; retraining/adaptiveselectionuncertaintyisnotcorrected.

Twofields take thelargestperiodogrampeak andtwo select itsperiod. Suchselectedextrema can bevariable evenwithnorealcycle; sixremainingfields averagepowerinpredefinedbands, plusonecoverageflag. A possible nextmechanismtest is anexplicitablationofpeak-selectionfields or calibrationofbanduncertainty—notsearchingmodel/weightparameters. No follow-uprecipeis admitted by thisreport alone.

All11inputcolumns,sourcecode,rawhashes,shuffledseed/permutationdefinition andtwentynewmodelsremainretained. NoKumo90prediction,creditarchitecture,hyperparameter,trainingbudget,seed,foldassignment or75:25weightchanged. Featurewidthchanges29/88 wereidentical fortrue/null. No confirmation orsubmissionopened/generated. Target.54remainsunmet; bestjointprovisionalcandidateisstillmaturity-only.530636/.847056,withuncertaintyincludingnogain.
