#!/bin/sh

for A in 0.01 # A values to use
do
mkdir A${A}/
cd A${A}/
apath='pwd'
for p in {8..24..2} # initial x-direction momentum
do
mkdir p${p}/
cd p${p}/
echo "p_vals=[$p]" | cat - ../../scan_template > input_${p}.tmp
echo "A_vals=[$A]" | cat - input_${p}.tmp > input_${p}.tmp.tmp
mv input_${p}.tmp.tmp input_${p}.tmp
path=`pwd`
echo "#!/bin/bash
# slurm commands can go here
cd ${path}
python3 /path/to/main.py ${path}/input_${p}.tmp
" > qsubmit_${p}.tmp
pwd
sbatch qsubmit_${p}.tmp
cd ..
done
cd ..

done