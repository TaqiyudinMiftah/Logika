set -eu
while systemctl --user is-active --quiet logika-transfer-core-maturity-20261025.service; do sleep 5; done
test -f CORE_MATURITY_RESULT.json
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python -u source/core_late30.py run --fold 2 > core_late30_fold2.log 2>&1
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python -u source/core_late30.py run --fold 4 > core_late30_fold4.log 2>&1
while ! test -f core_late30/fold3/complete.json; do sleep 5; done
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/core_late30.py report > core_late30_report.log 2>&1
