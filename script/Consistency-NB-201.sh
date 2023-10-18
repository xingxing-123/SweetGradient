export PYTHONPATH=./exp_consistency/NAS-Bench-101_201:$PYTHONPATH

path=logs/Consistency-NB-201
mkdir -p $path

dataset=cifar10 # cifar100, ImageNet16-120
datadir="data/cifar10"
gpu=0
seed=42
arch=100

# cal thr
CUDA_VISIBLE_DEVICES=${gpu} \
    nohup python -u exp_consistency/NAS-Bench-101_201/cal_THR1_THR2_nasbench2.py \
    --dataset $dataset --datadir $datadir --save_dir=$path --seed=$seed --end=$arch > ${path}/cal_thr.log 2>&1

# cal consistency
THR1=0.0001 # write thr here
THR2=0.0005 # write thr here
estimator="effective_capacity" # effective_capacity effective_gradnorm effective_snip effective_grasp effective_ntktrace effective_gradsign effective_zico
start=0
end=15625
batch_size=64
write_freq=1
seed=42 
outdir=$path

mkdir -p ${outdir}
outfname="${dataset}_arch[${start}-${end}]_measure[${estimator}]_seed${seed}"
CUDA_VISIBLE_DEVICES=${gpu[$i]} \
nohup python -u exp_consistency/NAS-Bench-101_201/nasbench2_pred.py \
    --outdir=${outdir} \
    --outfname=${outfname} \
    --measure_names=${estimator} \
    --start=${start} \
    --end=${end} \
    --batch_size=${batch_size} \
    --dataset=${dataset} \
    --datadir=${datadir} \
    --seed=${seed} \
    --THR1=${THR1} \
    --THR2=${THR2} \
    --write_freq=${write_freq} \
    > "${outdir}/${outfname}".log 2>&1

# check result
nohup python exp_consistency/NAS-Bench-101_201/check_201_result.py > ${outdir}/result.log
