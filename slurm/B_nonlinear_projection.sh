# Creates a copy of the repository and submits jobs to run the experiment
EXPERIMENT=B_nonlinear_projection
# First is template name (here same as experiment)
# Second is name of job
# Third is number of jobs
NUM_JOBS=$(python -m experiments.$EXPERIMENT.experiment_definition)
source slurm/utils/submit_experiment.sh $EXPERIMENT ExpB $NUM_JOBS
