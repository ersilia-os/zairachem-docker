#!/bin/bash

# Define the datasets
datasets=("AMES" "BBB_Martins" "Bioavailability" "hERG" "HIA_Hou" "Pgp_Broccatelli" "CYP2C9_Veith" "CYP2D6_Veith" "CYP3A4_Veith" "CYP2C9_Substrate_CarbonMangels" "CYP2D6_Substrate_CarbonMangels" "CYP3A4_Substrate_CarbonMangels")

# Loop over datasets
for dataset in "${datasets[@]}"
do
  # Loop over 5 folds for each dataset
  for i in {1..5}
  do
    # Run fit script for each dataset and fold
    bash run_fit.sh -i ../zaira-chem-docker-tdc/data/"${dataset}"_train.csv -m ../zaira-chem-docker-tdc/models/"${dataset}"_fold$i

    # Run predict script for each dataset and fold
    bash run_predict.sh -i ../zaira-chem-docker-tdc/data/"${dataset}"_test.csv -m ../zaira-chem-docker-tdc/models/"${dataset}"_fold$i -o ../zaira-chem-docker-tdc/predictions/"${dataset}"_pred_fold$i --flush True
  done
done
