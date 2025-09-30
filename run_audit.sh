#!/bin/bash

echo "Choose dataset to run audit (porto/beijing): "
read dataset

case "$dataset" in
  porto)
    echo "Running DP audit on Porto dataset."
    python -m DP_Audit.GRR.empirical_eps_porto &
    python -m DP_Audit.SS.empirical_eps_porto &
    python -m DP_Audit.UE.empirical_eps_oue_porto &
    wait
    ;;
  beijing)
    echo "Running DP audit on Beijing dataset."
    python -m DP_Audit.GRR.empirical_eps_beijing &
    python -m DP_Audit.SS.empirical_eps_beijing &
    python -m DP_Audit.UE.empirical_eps_oue_beijing &
    wait
    ;;
  *)
    echo "Invalid choice: $dataset. Please choose either 'porto' or 'beijing'."
    exit 1
    ;;
esac

echo "All done!"
