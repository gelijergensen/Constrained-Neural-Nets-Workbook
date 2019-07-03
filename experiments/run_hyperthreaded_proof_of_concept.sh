#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -J Hyper-POC
#SBATCH --mail-user=gejergensen@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# activate the virtualenv
source /global/u1/g/gelijerg/setup.sh
source activate torch-gpu
conda deactivate # I don't know the fuck why, but the first activation always goes slightly wrong on the cluster
source activate torch-gpu


#run the application:
srun -n 1 -c 272 --cpu_bind=cores python -u /global/u1/g/gelijerg/Projects/pyinsulate/experiments/hyperthreaded_proof_of_concept.py

