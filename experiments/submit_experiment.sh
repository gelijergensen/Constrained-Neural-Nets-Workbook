EXPERIMENT=$1

# First we do all necessary setup
source experiments/submitting_setup.sh

# Next we spawn a batch job with the right number of jobs
NUM_CONFIGS=$(python experiments/$EXPERIMENT/experiment_definition.py)
echo $EXPERIMENT with $NUM_CONFIGS configurations

cat experiments/submit_experiment.template | sed -e 's/$1/'$EXPERIMENT'/g' | sed -e 's/$2/'$(($NUM_CONFIGS - 1))'/g' > experiments/temp_$EXPERIMENT.sh
echo "Submission file looks like:"
cat experiments/temp_$EXPERIMENT.sh
echo ""
sbatch experiments/temp_$EXPERIMENT.sh
