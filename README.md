# zairachem-docker

To install ZairaChem v2 do the following:

``` 
git clone https://github.com/ersilia-os/zairachem-docker
cd zairachem-docker
bash install.sh
```
This will create a conda environment for each step of the processing pipeline.

## Model fitting

```
bash run_fit.sh -i [INPUT_FILE] -m [MODEL_DIR]
```
Optional flags for the fit command include:
```
-c|--cutoff <float>
-d|--direction high/low
-p|--parameters <parameters_file.json>
--clean True/False
--flush True/False
--anonymize True/False
```
If no cut-off and direction is passed the file must contain a column with the data already binarized

## Model predicting

```
bash run_predict.sh -i [INPUT_FILE] -m [MODEL_DIR] -o [OUTPUT_DIR]
```
Optional flags for the fit command include:
```
--clean True/False
--flush True/False
--anonymize True/False
```