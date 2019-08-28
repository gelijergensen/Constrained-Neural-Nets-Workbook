# The first argument to this script is the experiment and all subsequent will
# be used to rewrite the template

EXPERIMENT=$1

# First we do all necessary setup
source slurm/utils/submitting_setup.sh

# Next we determine the correct template file
TEMPLATE=slurm/templates/$EXPERIMENT.template

cat $TEMPLATE

# Loop over all passed arguments and use them to update the template
idx=0
for var in "$@"; do
  if [ "$idx" -eq "0" ] ; then
    SCRIPT="$(cat $TEMPLATE)"
  else
    # Add new variable: $(i+1) --> $i in template
    SCRIPT="$( echo "$SCRIPT" | sed -e 's/$'$idx'/'$var'/g' )"
  fi
  idx=$((idx+1))
done
echo "$SCRIPT" > slurm/temp_$EXPERIMENT.sh
echo "Submission file looks like:"
cat slurm/temp_$EXPERIMENT.sh
echo ""
sbatch slurm/temp_$EXPERIMENT.sh
