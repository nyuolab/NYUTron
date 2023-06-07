#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --qos=qos_free
#SBATCH --job-name=pretraining
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=lyj2002@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=40
#SBATCH --mem=800G
#SBATCH --partition=oermannlab
#SBATCH --time=30-00:00:00
#SBATCH --gres=gpu:a100:8
#SBATCH --output=outs/multinode_%j.out
#SBATCH --nodelist=a100-8001,a100-8002,a100-8003


echo "hostname:"
hostname

#source ~/.bashrc
#echo "setup env"
#conda init bash
source /gpfs/data/oermannlab/users/lavender/.bashrc
export HOME=/gpfs/home/jiangy09/
echo "home dir is"
echo $HOME
# ssh-copy-id -i ~/.ssh/id_rsa jiangy09@a100-8001 # copy ssh profile to new home dir
# echo "copying ssh key"

nvidia-smi
module load cuda/11.4 gcc/10.2.0 nccl
conda activate /gpfs/data/oermannlab/users/lavender/.conda/envs/ds_hf #ds_hf

which deepspeed

run_str='deepspeed --hostfile configs/hostfile --num_gpus=8 --num_nodes=3 pretrain_multinode_hydra.py --deepspeed configs/pretrain_configs/deepspeed_config_multinode.json'
echo "$run_str"
$run_str

# in case deepspeed error out
# reference: https://huggingface.co/docs/transformers/main_classes/deepspeed
# check cuda architecture with 
# CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
# new installation with
# TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
# --global-option="build_ext" --global-option="-j8" --no-cache -v \
# --disable-pip-version-check 2>&1 | tee build.log
