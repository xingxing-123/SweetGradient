export PYTHONPATH=./exp_search/NAS-Bench-201:${PYTHONPATH}

dataset="cifar10"
data_path="data/cifar10"
space=nas-bench-201
benchmark_file="data/NAS-Bench-Data/NAS-Bench-201-v1_0-e61699.pth"
score='effective_capacity'

gpu=0
batch_size=64
n_runs=5
seed=-1
channel=16
num_cells=5
max_nodes=4

save_dir=logs/Search-NB-201
mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=${gpu} \
OMP_NUM_THREADS=4 python -u exp_search/NAS-Bench-201/my_EA.py \
    --score ${score} \
    --batch_size ${batch_size} \
    --data_path ${data_path} \
    --save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
    --dataset ${dataset} \
    --search_space_name ${space} \
    --arch_nas_dataset ${benchmark_file} \
    --time_budget 99999 \
    --n_runs $n_runs \
    --ea_cycles 500 --ea_population 25 \
    --workers 1 --rand_seed ${seed} \
    > $save_dir/${score}_bs${batch_size}.log 2>&1

