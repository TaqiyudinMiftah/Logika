set -eu
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/early_average.py run --fold 0 > early_average_fold0.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/early_average.py run --fold 1 > early_average_fold1.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/early_average.py run --fold 2 > early_average_fold2.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/early_average.py run --fold 3 > early_average_fold3.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/early_average.py run --fold 4 > early_average_fold4.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/early_average.py report > early_average_report.log 2>&1
