#!/bin/bash
#SBATCH --job-name=SDialogIndexTTS+dScaper+RoomAccoustics+StaticRooms
#SBATCH -C h100
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=8           # nombre de CPU par tache (un quart du noeud ici)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=./logs_slurm_static_rooms/SDialogIndexTTS-%j.out # nom du fichier de sortie
#SBATCH --error=./logs_slurm_static_rooms/SDialogIndexTTS-%j.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --account=rtl@h100
#SBATCH --qos=qos_gpu_h100-t3

module purge

module load arch/h100
module load cuda/12.4.1
module load ffmpeg/6.1.1

module load miniforge

conda activate jsalt10

set -x -e

export OMP_NUM_THREADS=8

export CUDA_LAUNCH_BLOCKING=1

export NCCL_ASYNC_ERROR_HANDLING=1

srun -l python generate_audios_jean_zay+dscaper+accoustics+static_rooms.py --worker_id=0 --nbr_worker=1
