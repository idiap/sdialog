#!/bin/bash
#SBATCH --job-name=build_sdialog_apptainer
#SBATCH --output=build_sdialog_apptainer_%A_%a.out
#SBATCH --error=build_sdialog_apptainer_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --nodes 1
#SBATCH --time=1:30:00
#SBATCH --gpus rtx3090:1
#SBATCH --account YOUR_ACCOUNT

OUTPUT_FILE="sdialog.sif"

# build apptainer
apptainer build --fakeroot $OUTPUT_FILE Apptainer

echo "Finished building $OUTPUT_FILE"
