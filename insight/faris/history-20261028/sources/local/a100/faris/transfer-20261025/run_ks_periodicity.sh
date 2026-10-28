set -eu
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/ks_periodicity.py run --fold 0 > ks_periodicity_fold0.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/ks_periodicity.py run --fold 1 > ks_periodicity_fold1.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/ks_periodicity.py run --fold 2 > ks_periodicity_fold2.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/ks_periodicity.py run --fold 3 > ks_periodicity_fold3.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/ks_periodicity.py run --fold 4 > ks_periodicity_fold4.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/ks_periodicity.py report > ks_periodicity_report.log 2>&1
