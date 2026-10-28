# Logika six-hour broad campaign

The new immutable clock started2026-10-28 17:25:40UTC and ends23:25:40UTC (29October06:25:40WIB). Admissions close22:40:40UTC and model fitting ends23:05:40UTC. Never reset or extend it.

Canonical state: faris@100.68.29.126:/mnt/home/faris/logika-sixhour-broad-20261028/LIVE_STATE.json; coordinator logika-sixhour-coordinator-20261028.service. Lab prep, CPU tasks and node2003 spillover are owned user services. No old campaign is restarted.

Initially granted account2 sessions were4=L4,5=L4,6=A10080GB. VM5 completed all130owned fits and was released after local/canonical output hash verification on28October19:11UTC. Only4and6 remain active; never recreate5. AdditionalA100 requests2/3 and fourthconcurrentL4 request7 were refused. VM1 was released after SSHlaunchacktimeout; its completedTabMpilot is retained and2interruptedKumo attempts have explicit retry2. Do not restoreVM1. Workstation ops/colab_control_v2.py PIDfiles private/vmN/CONTROLLER_PID manage only4/5/6. Whole-design task ownership is inTASK_OWNERS.json. Each notebook kernel also has an independent deadline release thread.

CPU folds0-2 run on2001 and3-4 on2003, only under actual /mnt/home/faris. No retired2002 or unrelated workload interference. Runtime pilots qualify each family before broad admission. Model/data/feature scientific source is sealed; operational repairs are logged separately.

64initial plus36CPUcontrols and8separateFP32ResNetreplications,2seeds and5folds plan1080fits; this is not a completion claim. Candidate feature bank2129 includes39new sourced actuarial candidates; reduced32-512,99-feature Kumo and64PCA+20core views are compared. All statistics/selectors/PCAfit42500inner rows,7500stopper excluded,50kcontext modelrefits and6kheldmeta,14kouter. All70k rows are adaptively reused development, no untouched audit. Complete10fit cohorts only; finalchoice frozen before conditional uncertainty. PublicV25 .56822 is separate frommatchedlocal .5306361042400267.

Every attempt, includingnegative/failed/partial, is published sanitized to insight/faris/runs/sixhour-broad-20261028 insharedLogika. Credentials/rawdata/case-level labels/predictions private; no fitted model or KV dumps. No Kaggle submissions. The book PDF/text remainprivate with public metadata/researchreferences.

30minute chatfollowup is logika-sixhour-followup-20261028, ACTIVE via app automation tool. Atcompletion collect/audit evidence, verify allownedVMs released, report andpause it.

Execution repairs: CPUvenv dependencies installed explicitly, dispatch retry bounded byAPPROVED_RETRIES. Compatibilityops/queue.py exports stdlibqueue when imported and launches worker_queue.py only asCLI, avoiding module shadowing. The sealedworker omittedGPU_name/memory fields expectedby gate; ops/qualify_gpu.py resolves only those measured fields from same-sessionGPU_BACKENDS evidence, preserves all original checks andfailedgateattempts. No START, model/features/source seal, contract orCLOCK changed. Qualifier_loopservice handles thoseGPUqualifications.

FP32stabilityreplications use documented NVIDIA_TF32_OVERRIDE=0 declared in eachnewrecipe andprelaunchBACKEND, original5e-6replay check unchanged. Source-defaultTF32failures remainunscored numerical limitations. FirstnewFP32foldpassed. Composeadapter also coordinates evaluate.lock; finalGPUcollection flag waitsfor terminalqueues plusfinalcompositionbarrier beforefrozenchoice. All7familygates nowPASS via originalchecks andmeasuredGPUmetadataadapter.

Developmentevaluation helper: logika-sixhour-evaluation-20261028.service invokessealedmathematics throughops/evaluate_blocking.py. Oldcoordinator evaluation.log maycontain BlockingIOError whilehealthycomposition holdsevaluate.lock; inspectlogs/evaluation_blocking.log instead. Do not restarthealthycoordinator/workers becauseoflockcontention. Periodichelperstops15minutes beforemodeldeadline; finalbarrier releaseslock beforeterminalselection.

Storage execution repair on19:01heartbeat: see STORAGE_REPAIR.json. Private source/input bundles were hash-verified and relocated with original paths preserved; only4abandoned temporaryGitpacks (classified as garbage, not committed objects) were removed. Backup/publication resumed. Publisher disables implicit maintenance and links identical terminal private evidence to existing incremental files. Backup reserves512MiBfree. VM5output/release audit is private/vm5/TERMINAL_COLLECTION_AUDIT.json. No scientific seal or clock changed.
