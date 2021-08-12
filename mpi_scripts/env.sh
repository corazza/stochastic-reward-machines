#!/bin/bash
today=`date '+%Y_%m_%d__%H_%M_%S'`
hourly=`date '+%Y_%m_%d__%H'`
exp_name=${1:-${hourly}}
outer_results="../mpi_results/${exp_name}"
results_dir="${outer_results}/${envalg_name}_${today}"

mkdir -p $outer_results
mkdir $results_dir
