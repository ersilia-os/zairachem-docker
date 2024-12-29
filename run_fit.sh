eval "$(conda shell.bash hook)"
conda activate zairasetup
python 01_setup/zairasetup/run_fit.py -i ../tests/osm_bin.csv -m ../tests/fit

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