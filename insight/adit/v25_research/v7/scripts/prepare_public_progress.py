"""Prepare reviewable, credential-free progress for the shared GitHub repo."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'v7'
DEST=ROOT.parent/'github_sync_20261027'/'insight/adit/v25_research'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    master=json.loads((V/'reports/EXPERIMENT_MASTER_LOG.json').read_text(encoding='utf-8'))
    planned=master['active_experiments']
    experiments=[x for x in planned if x['status']=='COMPLETE' and x.get('result')]
    modeling=[x for x in experiments if int(x['id'][1:])>=120]
    assert len(modeling)>=40 and all(x['status']=='COMPLETE' for x in modeling)
    (DEST/'EXPERIMENT_QUEUE.json').write_text(json.dumps([dict(id=x['id'],status=x['status'],definition=x['definition']) for x in planned if x['status']!='COMPLETE'],indent=2),encoding='utf-8')
    registry=[];table=[]
    for x in experiments:
        r=x['result'];signature=dict(configuration=r['spec'],features=r['columns'],context_features=r.get('context_columns'),feature_matrix_sha256=r.get('feature_matrix_sha256'),extension_feature_matrix_sha256=r.get('extension_feature_matrix_sha256'),folds=[dict(fold=f['fold'],validation_ids_sha256=f['validation_ids_sha256'],training_ids_sha256=f.get('train_ids_sha256')) for f in r['folds']],backbone='BASELINE_LOCK.json')
        canonical=dict(signature,configuration={k:v for k,v in signature['configuration'].items() if k not in ['id','label','families','only','fold_feature_root','feature_catalog','extension','discovery','repayment','statistical','behavior','lifecycle','fusion','combination','modeling']})
        fingerprint=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        registry.append(dict(id=x['id'],signature_sha256=fingerprint,configuration=signature,KS=r['OOF']['KS'],AUC=r['OOF']['AUC'],comparison_baseline=r.get('comparison_baseline','V00'),delta_KS=r['delta_KS'],delta_AUC=r['delta_AUC'],public_observations=x['public_leaderboard_observations'],submission=r['submission'],submission_sha256=r['submission_sha256'],node_result_source=f"v7/runs/{x['id']}/RESULT.json",original_result_sha256=x['source_result_sha256']))
        table.append(dict(experiment=x['id'],method=x['definition']['label'].replace('QUARANTINED ',''),KS=r['OOF']['KS'],AUC=r['OOF']['AUC'],comparison_baseline=r.get('comparison_baseline','V00'),delta_KS=r['delta_KS'],delta_AUC=r['delta_AUC'],fold_wins=r['fold_wins'],delta_KS_CI_low=r['confidence_interval']['delta_KS'][0],delta_KS_CI_high=r['confidence_interval']['delta_KS'][1],head_gap=sum(f['train_validation_head_KS_gap'] for f in r['folds'])/5,formula=r.get('modeling_formula',x['method_formula']),submission_sha256=r['submission_sha256']))
        if int(x['id'][1:])>=120 or x['id'] in ['V00','V25','V118','V119']:
            folder=DEST/'results'/x['id'];folder.mkdir(parents=True,exist_ok=True)
            (folder/'RESULT_VIEW.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
            for name in ['ERROR_ANALYSIS.json','ERROR_ANALYSIS.md','ERROR_SEGMENTS.csv','REPORT.md','MATCHED_CONTROL.json','MODELING_METHOD.json','GPU_CACHE_PARITY.json','PROCESS_ATTEMPTS.json','PREVIOUS_ATTEMPT_FAILURE.json','CONTEXT_INPUT_AUDIT.json','REQUIREMENTS_AUDIT.json']:
                source=V/'runs'/x['id']/name
                if source.exists():shutil.copyfile(source,folder/name)
    (DEST/'EXPERIMENT_REGISTRY.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
    with (DEST/'EXPERIMENTS.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    catalogs=[]
    for source,rows in master['all_active_feature_catalogs'].items():
        catalogs.extend(dict(source_catalog=source,**row) for row in rows)
    keys=list(dict.fromkeys(k for r in catalogs for k in r))
    with (DEST/'FEATURE_CATALOG.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(catalogs)
    files=['BASELINE_LOCK.json','V25_BASELINE_LOCK.json','ACTIVE_BASELINE.json','requirements-lock.txt','scripts/baseline.py','v7/modeling_plan.json','v7/V25_MODELING_PLAN.md','v7/FEATURE_POLICY.json','v7/scripts/modeling_features.py','v7/scripts/modeling_representations.py','v7/scripts/modeling_experts.py','v7/scripts/modeling_ranking.py','v7/scripts/experiments.py','v7/scripts/features.py','v7/scripts/feature_policy.py','v7/scripts/master_log.py','v7/scripts/prepare_public_progress.py','v7/scripts/sync_completed_experiment.py']
    for name in files:
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    for name in ['v7/scripts/basket_value_features.py','v7/scripts/cache_spectral_features.py','v7/light_fe/reports/BASKET_VALUE_BUILD.json','v7/light_fe/reports/SPECTRAL_CACHE.json','v7/kumo_plan.json','v7/scripts/setup_kumo_gpu.py','v7/scripts/kumo_gpu_experiment.py','v7/scripts/kumo_gpu_experiment_v2.py','v7/scripts/supervise_kumo.py','v7/scripts/continue_frozen_kumo_queue.py','v7/KUMO_FROZEN_QUEUE.md','v7/kumo_gpu/WEIGHTS_MANIFEST.json']:
        if (ROOT/name).exists():
            target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    for name in ['v7/reports/MODELING_QUEUE.json','v7/modeling/reports/CACHE_AUDIT.json','v7/modeling/reports/GRADIENT_CHECK.json','v7/modeling/reports/MODELING_EXPERIMENTS.csv']:
        if (ROOT/name).exists():
            target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    for name in ['v7/KUMO_QUEUE_10.json','v7/KUMO_QUEUE_10.md','v7/KUMO_EXTRA_QUEUE.md','v7/KUMO_EXTRA_REJECTED_DRAFT.json','v7/scripts/kumo_gpu_experiment_v3.py','v7/scripts/cached_frozen_experiment.py','v7/scripts/replay_v25_test_cache.py','v7/scripts/audit_frozen_result.py','v7/reports/V25_TEST_CACHE_REPLAY.json','v7/scripts/extra_frozen_context.py','v7/scripts/review_extra_frozen.py','v7/kumo_gpu/extra_review/REVIEW.json','v7/kumo_gpu/extra_review/DRAFT_REVIEW.json']:
        if (ROOT/name).exists():
            target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    best_result=max(modeling,key=lambda x:x['result']['OOF']['KS'])['result']
    for name in ['v7/conservative_plan.json','v7/CONSERVATIVE_QUEUE_5.md','v7/KUMO_QUEUE_15.json','v7/KUMO_QUEUE_15.md','v7/scripts/conservative_adapters.py','v7/conservative/reports/GRADIENT_CHECK.json','v7/conservative/reports/LIGHTGBM_OFFSET_CHECK.json']:
        if (ROOT/name).exists():
            target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    latest_result=max(modeling,key=lambda x:int(x['id'][1:]))['result']
    selected=[next(x['result'] for x in experiments if x['id']=='V25'),best_result,latest_result]
    for selected_result in selected:
        filename=Path(selected_result['submission']).name
        source=V/'submissions'/filename
        if source.exists():
            target=DEST/'submissions'/filename;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
    best=max(modeling,key=lambda x:x['result']['OOF']['KS'])['result']
    ci=best['confidence_interval']['delta_KS']
    report=f'''# Riset V25: target pooled OOF KS0.531

Baseline public tetap **V25:0.56822**, verified terhadap hash CSV. Pooled OOF V25 **0.5302846600**, AUC **0.8470292789**. Target pooled0.531 {'tercapai pada fold pengembangan; generalisasi belum terbukti' if best['OOF']['KS']>=.531 else 'belum tercapai'}. Tidak ada submission otomatis ke Kaggle.

**{len(experiments)} eksperimen aktif tercatat**, termasuk{len(modeling)} eksperimen modeling baru. Semuanya memiliki lima fold, OOF, checkpoint dan submission di workspace node480. Hasil terbaik baru **{best['experiment']}: KS{best['OOF']['KS']:.10f}, AUC{best['OOF']['AUC']:.10f}**, deltaKS vsV25 **{best['delta_KS']:+.10f}**, dengan{best['fold_wins']}/5 fold menang. Paired95% deltaKS CI **[{ci[0]:+.8f},{ci[1]:+.8f}]**. Kandidat tidak otomatis menggantikan incumbent public.

## Hasil dan arah penelitian

- V120:V131 menguji regularisasi, kekuatan koreksi dan probability/logit blend. V127 menurunkan L2 dari0.003 ke0.001 dengan strength0.5 dan blend75:25 tetap.
- V132:V143 menguji pasangan fitur dan embedding interaksi rank4/8 pada18 input V25 yang sama. Semua lebih rendah daripada V25; memperbesar representasi pada6k correction rows belum membantu.
- V144:V151 menambahkan dua histogram-GBDT, dilatih56k outer-training dengan inner validation. Input116 adalah union informasi yang telah dipakai pipeline V25, tanpa fitur histori baru. Expert standalone KS0.52154/0.52115; fixed blends terbaik0.530528, belum melebihi V127.
- V152:V159 menguji pairwise ranking dan smooth-CDF loss yang ditambatkan ke V25. Semua lebih rendah daripada V25. Eksperimen terakhir V159 KS0.5301448252, AUC0.8470188808.

Setiap eksperimen memiliki `ERROR_ANALYSIS.md/json`, `ERROR_SEGMENTS.csv`, confidence interval dan metrik per-fold di `results/`. Analisis menggunakan prediksi outer-held-out, mencatat perbaikan maupun kemunduran top-risk capture dan segmen. Case-level OOF/target, raw data dan checkpoint besar tetap di node480, tidak dipublikasikan.

## Reproduksi

Workspace runtime: `/home/fariselhakim/adit/Logika/insight/goal054_offset_baseline_20261019`.
Lingkungan: `/home/fariselhakim/adit/.venv-logika-offset-20261027/bin/python`.
Kode ada di `v7/scripts/`; formula/metode tercatat di `EXPERIMENTS.csv`, konfigurasi dan fingerprint di `EXPERIMENT_REGISTRY.json`. Jalur `data/`, `artifacts/`, `v7/extension/data/` membutuhkan cache privat yang sudah diaudit, bukan diunduh dari GitHub. Jangan menjalankan ulang batch sebelum memeriksa protokol sinkronisasi.

Kumo Large100% frozen,90 input,50k context,8 estimators. Koreksi V25 memakai18 input dan6k rows; original spline77; lima outer fold seed31027 identik. Target/model selection memakai fold pengembangan berulang; CI bootstrap bersyarat pada fitted scores, tidak mengoreksi pencarian atau retraining. Ketidakpastian snapshot DPD/global references lama tetap diwarisi.

## Kolaborasi

Baca `SYNC_PROTOCOL.md` sebelum setiap eksperimen. Fetch main terbaru, periksa hasil teman, bandingkan signature, jalankan satu eksperimen, catat hasil/error analysis/submission hash, lalu push fast-forward. Jangan menimpa hasil teman atau memakai force push.
'''
    (DEST/'README.md').write_text(report,encoding='utf-8')
    with (DEST/'README.md').open('a',encoding='utf-8') as f:
        f.write(f"\n\nLatest completed: **{latest_result['experiment']}**, pooledKS **{latest_result['OOF']['KS']:.10f}**, AUC **{latest_result['OOF']['AUC']:.10f}**. See its results directory.\n")
    protocol='''# Protokol eksperimen bersama

1. Sebelum SETIAP eksperimen, baca HEAD terbaru main dan perubahan sejak receipt terakhir. Jika ada commit teman, baca konfigurasi, definisi fitur, folds, hasil, AUC dan error analysis terlebih dahulu.
2. Bandingkan signature konfigurasi+fitur+data/fold/backbone dalam EXPERIMENT_REGISTRY.json. Nama eksperimen berbeda belum membuktikan metode berbeda. Tandai duplikat dan skip; replication/seed test harus diberi alasan eksplisit.
3. Jalankan satu eksperimen pada fold yang sama. Preprocessing, early stopping, ranking pairs dan learned representation tidak boleh memakai outer-validation labels.
4. Simpan hasil termasuk negatif, OOF, submission, kode/checkpoint/formula, matched CI dan analisis error. Data/OOF berlabel dan checkpoint besar tetap lokal/node; GitHub mendapat kode, laporan agregat, metadata dan hash.
5. Fetch HEAD sekali lagi sebelum push. Buat commit di atas HEAD terbaru dan update main dengan expected SHA serta force=false. Bila teman keburu push, baca perubahannya dan bangun commit baru di atas head baru; jangan force.
6. Setelah push verifikasi commit dan file, tulis receipt pada node, lalu evaluasi eksperimen berikutnya.

Limit RAM16GiB dibagi semua akun pada kontainer. Periksa headroom dan peak proses sebelum training. Jangan memulai model berat ketika ruang terlalu kecil. Jangan mengakhiri proses pengguna lain. V25 tetap incumbent public sampai ada bukti baru yang jelas.
'''
    (DEST/'SYNC_PROTOCOL.md').write_text(protocol,encoding='utf-8')
    (DEST/'AGENTS.md').write_text('''# Shared V25 research workspace

Use V25 as baseline. Follow SYNC_PROTOCOL.md before and after EACH experiment, including checking peer commits and redundancy. Keep Kumo100% frozen. The user allows modeling and feature research to reach pooled KS>=0.531, but validation isolation and truthful uncertainty remain required. Preserve all negative outcomes. Publish no credentials, raw data, case-level labels/OOF or large model files. Never overwrite peer results or force push. Do not restart the entire historical queue automatically; run one new experiment, report, then sync. Runtime private cache lives on node480.
''',encoding='utf-8')
    manifest=[]
    for path in sorted(DEST.rglob('*')):
        if path.is_file():manifest.append(dict(path=path.relative_to(DEST).as_posix(),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (DEST/'PUBLIC_MANIFEST.json').write_text(json.dumps(dict(prepared_UTC=datetime.now(timezone.utc).isoformat(),files=manifest,credential_files_included=False,raw_training_rows_included=False),indent=2),encoding='utf-8')
    print(json.dumps(dict(destination=str(DEST),files=len(manifest)+1,bytes=sum(x['bytes'] for x in manifest),experiments=len(experiments),modeling_experiments=len(modeling)),indent=2))


if __name__=='__main__':main()
