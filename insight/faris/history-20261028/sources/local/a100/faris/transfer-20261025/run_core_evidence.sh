set -eu
while ! test -f CORE_MATURITY_RESULT.json; do sleep 20; done
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/audit_maturity_placement.py --hypothesis H006 --require-complete
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/core_uncertainty.py --hypothesis H006
while ! test -f CORE_LATE30_RESULT.json; do sleep 20; done
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/audit_maturity_placement.py --hypothesis H007 --require-complete
/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python source/core_uncertainty.py --hypothesis H007
