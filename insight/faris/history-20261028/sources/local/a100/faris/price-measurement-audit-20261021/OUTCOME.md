# Price-floor/rounding hypothesis not admitted to credit-model training

Originalfold0context-only audit: 2178879itemrows,1493075orders,50000customers. Alltransactions predate decisions, quantities arepositive, itemIDsunique, and everyunitprice ison the1000IDRgrid.

Only314items (.014411%) areat5000IDR, affecting311customers (.622%). Among affectedcustomers, the medianflooritemshare is2.174%; only3customers haveflooritems inmorethanoneorder. This isinsufficient evidence for prioritizing a broad censored-price reconstruction under the current goal. A loweredgepileup doesnotprove theoriginalgenerator or a particular censoringrule.

Conditional on nearest1000rounding and excludingthe314floorrows, the average squared log-price errorenvelope is.0000897468. This is.0210% of theexistingfitteditemnoisevariance. An illustrativeuniformwithinintervalvariance is.00684% of that itemnoisevariance. These are observation-scale comparisons underanassumedroundinginterpretation, notbounds onKS orproofthatroundedvaluescannotmatter. Theoriginalgenerator and floorlatentvalues remainunknown.

No featurevalue, probability, model, validationfold ortrainingparameter waschanged. No classifierfit wasadmitted, so this audit isnotcounted as a completedpredictiveexperiment. It removes anunsupportedpotentialdirection and preserves the evidence ratherthancreatinganother modelconfiguration. Source, exactcounts, percategory/statusbreakdown and numericalenvelope code are retained inRESULT.json,category_status_counts.csv,customer_floor_counts.parquet,ROUNDING_SCALE.jsonand source/.

Theexistingpricefactory already models item/ordermeasurementvariance and borrowerempirical-Bayesshrinkage. A future price-estimation proposal must identify a larger, independentlysupported misspecification ratherthanrepeat the floorstory. Defaultlabels/outer-customerhistories were notread. Confirmationclosed,nosubmission,target.54stillunmet.
