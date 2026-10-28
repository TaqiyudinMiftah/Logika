# Conditional account measurement replaces existing income summaries

Prior evidence: context-only raw-observation audit withheld the last3complete months of each originalfold0contextcustomer. AnAR1measurement model plus borrower randomintercept reduced log-incomeMSE .2143176->.1973506 and log-outflowMSE .3043091->.2550489 relative to its iid-noise control. Pairedcustomerintervals excluded zero; individualGaussianlikelihood and directconditionalforecast checks passed. This is not default-scoring evidence. Previously added trend columns and sourcecounts were negative and are not reused as candidate additions here.

One feature hypothesis: the existing long-history geometric means can lag current observed financial state because monthly observations are serially dependent. Replace them with a one-stepconditional expected log observation given allavailable completed months, using the same bounded Gaussian measurement estimator. It estimates log-income/outflows in the next observed-calendar month; it is not a recovered latent true household income. Outflow bias remained in the raw diagnostic, so this model is approximate.

Fit separate income/outflowcovariances inside each original50kcontext using all its availablecomplete histories. Gaussianparameters: equal-customer logmean, borrowerbetweenvariance, AR1stationarymeasurementvariance/correlation. FixedL-BFGS-B100iterations and bounds from the audit; no default label or queryhistory estimates a shared parameter. Transform eachapplicant's own history afterward. Missing source remains missing. No extra field dimension is introduced into a predictor.

Replace exactly5existing aliases where consumed:

* re_income_geometric_mean = exp(conditional nextmonth logincome mean).
* re_income_log_mean = that conditional logmean.
* re_declared_income_log_gap = log(declaredmonthlyincome) minusconditional logincome mean.
* re_outflow_income_geometric_ratio = exp(conditional logoutflow mean minusconditional logincome mean), a geometricpoint ratio, not E[outflow/income].
* re_income_log_measurement_sd = conditional nextmonth logincome observationSD. This changes uncertainty semantics from theoldiid-averageSE; lineage explicitly distinguishes them.

The correction consumes3ofthese5; splineconsumesall5. Retain historical logdispersion and allotherinputs. The two maturity-aligned repaymentposterior means from the provisionallead remain fixed. Kumo90andrawpredictionsunchanged. Architecture18-inputhead/77-inputspline, head6kL2.003/strength.5, spline56kC.05/fiveknots/degree3/newton-cholesky/maxiter80/tol1e-5, seeds/folds/budgetsand75:25blend unchanged. Fit only5newheads+5newsplines; serialize/replayandretainall. No hyperparameter/model/weightsearch, no horizonchoice, no partialfoldscoring.

Compare full70kOOFKS/AUC/logloss againstoriginalrepairedV25 andmaturity-only. Reportperfoldchanges, actualtimings, source/hashlineage andconditionalpairedbootstrap1000replicationswith2contrasts x2metricsBonferroni95percent. Require positiveKS/AUC andfoldconsistency beforeprovisionalpromotion. Historicalsearch/retraininguncertainty remains; no untouchedgeneralizationclaim. Confirmationstaysclosed,nosubmission. Target.54unmet.

Originaldeadline1792585871.823748/admissionstop3hoursbefore remains. Featuremodelcode is copied/hash-locked from the completed source-observationaudit. No othernewrecipeis admitted whilethisonecompletes.
