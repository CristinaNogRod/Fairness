#!/bin/bash

function help
{
   # Display Help
   echo "This script runs a given experiment N times and stores all the produced reports."
   echo "Those reports can be further analyzed for getting statistics"
   echo
   echo "Syntax: run_test.sh <experiment_id> [num_runs]"
   echo "Examples:"
   echo "Upload a file:"
   echo "  run_test.sh adult 10"
   echo
   echo "options:"
   echo "--h     Print this Help."
   echo
}

while getopts ":h" option; do
   case $option in
      h) # display Help
         help
         exit;;
   esac
done


if [ $# -eq 0 ]; then
    echo "No arguments provided. Check available options with --h."
    exit 1
fi

runs=$2
experiment=$1


if [ $2 -eq 0 ]; then
    runs=1
fi


for (( c=1; c<=$runs; c++ ))
do  
    #python "src/experiments/${experiment}.py"
    #mv "outputs/report_${experiment}.json" "outputs/report_${experiment}_${c}.json"
    echo $c
done