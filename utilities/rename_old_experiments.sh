#!/bin/bash
# copy files from old experiments structure with subfolders to new flat structure
experiment="Stepper-Decay_old"
out_dir="Stepper-Decay"
mkdir $out_dir
cp $experiment/R00/*.grammar $out_dir
cp $experiment/R00/*.json $out_dir
for i in $experiment/R*; do
arr_i=(${i//// })
run_number=${arr_i[1]//R/r}
echo "RUN $run_number DIR $i"
cp $i/*.log $out_dir
array=("best_parent.pkl" "population.pkl" "statistics.pkl" "gen_50_population.pkl" "gen_50_statistics.pkl" "gen_100_population.pkl" "gen_100_statistics.pkl" "gen_150_population.pkl" "gen_150_statistics.pkl")
for file in "${array[@]}"; do
    with_run="${run_number}_$file"
    outfile=${with_run//gen_/gen}
#    echo $file $outfile
    cp $i/$file $out_dir/$outfile
done
done
