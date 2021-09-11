#!/bin/bash
today=`date '+%Y_%m_%d__%H_%M_%S'`
hourly=`date '+%Y_%m_%d__%H'`
outer_results="../results/${exp_name}"
results_dir="${outer_results}/${today}"

mkdir -p $outer_results
mkdir $results_dir
