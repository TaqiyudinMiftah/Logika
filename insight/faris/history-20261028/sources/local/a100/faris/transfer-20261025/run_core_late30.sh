set -eu
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python -u source/core_late30.py run --fold 0 > core_late30_fold0.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python -u source/core_late30.py run --fold 1 > core_late30_fold1.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python -u source/core_late30.py run --fold 2 > core_late30_fold2.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python -u source/core_late30.py run --fold 3 > core_late30_fold3.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python -u source/core_late30.py run --fold 4 > core_late30_fold4.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/core_late30.py report > core_late30_report.log 2>&1
