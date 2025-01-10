#!/bin/bash

# Define the datasets
datasets=("DILI" "hERG" "CYP3A4_Veith")

# Loop over datasets
for dataset in "${datasets[@]}"
do
  # Loop over 5 folds for each dataset
  for i in {1..5}
  do
    # Run fit script for each dataset and fold
    bash run_fit.sh ../zaira-chem-docker-tdc/data/"${dataset}"_train.csv ../zaira-chem-docker-tdc/models/"${dataset}"_fold$i

    # Run predict script for each dataset and fold
    bash run_predict.sh ../zaira-chem-docker-tdc/data/"${dataset}"_test.csv ../zaira-chem-docker-tdc/models/"${dataset}"_fold$i ../zaira-chem-docker-tdc/predictions/"${dataset}"_pred_fold$i
  done
done
