eval "$(conda shell.bash hook)"
conda activate zairasetup
python 01_setup/zairasetup/run_predict.py -i "$1" -m "$2" -o "$3"

conda activate zairadescribe
python 02_describe/zairadescribe/run.py

conda activate zairatreat
python 03_treat/zairatreat/run.py

conda activate zairaestimate
python 04_estimate/zairaestimate/run.py

conda activate zairapool
python 05_pool/zairapool/run.py

conda activate zairareport
python 06_report/zairareport/run.py

conda activate zairafinish
python 07_finish/zairafinish/run.py