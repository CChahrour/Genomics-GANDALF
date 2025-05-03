#!/bin/bash
# This script is used to train the GANDALF_SEM model on the MLL-N dataset with different region names.
python train_best.py --data_dir /home/ubuntu/project/data --region_name promoters_1024bp --target MLL-N --project SEM_MLL-N_TF --model_name GANDALF_SEM --best_config_path results/SEM_MLL-N_TF/GANDALF_SEM_promoters_1024bp_MLL-N_singlelabel_regression_2025-04-26_1556/best_run_config.json
python train_best.py --data_dir /home/ubuntu/project/data --region_name methylome_1024bp --target MLL-N --project SEM_MLL-N_TF --model_name GANDALF_SEM --best_config_path results/SEM_MLL-N_TF/GANDALF_SEM_methylome_1024bp_MLL-N_singlelabel_regression_2025-04-26_1644/best_run_config.json
