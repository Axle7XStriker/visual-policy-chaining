#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32GB
#SBATCH --output=/home1/amanbans/csci566-project/job.log
#SBATCH --account=jessetho_1016 

#SBATCH --mail-user=amanbans@usc.edu
#SBATCH --mail-type=ALL

module purge
module load gcc/11.3.0 git/2.36.1 libx11/1.8 xproto/7.0.31 glew/2.2.0 mesa-glu/9.0.2 patchelf/0.14.5 cuda/11.6.2 mesa/22.3.2 openmpi micro cudnn/8.4.0.27-11.6

eval "$(conda shell.bash hook)"
conda activate bellman

#jupyter nbconvert --execute ./1/Problem_2.ipynb --to notebook --output Problem_2.ipynb
#jupyter nbconvert --execute ./1/Problem_2.ipynb --to notebook --inplace --debug

# add mujoco to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# for GPU rendering
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export CPATH=$LIBX11_ROOT/include:$XPROTO_ROOT/include:$GLEW_ROOT/include:$MESA_GLU_ROOT/include:$MESA_ROOT/include:$CPATH
export LDFLAGS='-L$MESA_ROOT/lib'

export GPUS=$SLURM_JOB_GPUS
printenv $GPUS

# Sub-task demo generation
python -m furniture.env.furniture_sawyer_gen --furniture_name chair_ingolf_0650 --demo_dir /scratch1/amanbans/bellman/demos/chair_ingolf/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --start_count 0 --phase_ob True --visual_ob True
#python -m furniture.env.furniture_sawyer_gen --furniture_name chair_ingolf_0650 --demo_dir /scratch1/amanbans/bellman/demos/chair_ingolf/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --preassembled 0 --start_count 1000 --phase_ob True --visual_ob True
#python -m furniture.env.furniture_sawyer_gen --furniture_name chair_ingolf_0650 --demo_dir /scratch1/amanbans/bellman/demos/chair_ingolf/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --preassembled 0,1 --start_count 2000 --phase_ob True --visual_ob True
#python -m furniture.env.furniture_sawyer_gen --furniture_name chair_ingolf_0650 --demo_dir /scratch1/amanbans/bellman/demos/chair_ingolf/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --preassembled 0,1,2 --start_count 3000 --phase_ob True --visual_ob True

# Full-task demo generation
#python -m furniture.env.furniture_sawyer_gen --furniture_name chair_ingolf_0650 --demo_dir demos/chair_ingolf/ --reset_robot_after_attach True --max_episode_steps 800 --num_connects 4 --n_demos 200 --start_count 0 --phase_ob True

#export WANDB_API_KEY=8deb5f09eadda27705889f1127157d752948cad6
# Train sub-task policies
#mpirun -np 16 python -m run --algo gail --furniture_name chair_ingolf_0650 --phase_ob True --visual_ob True --demo_path /media/aman-anmol/Aman/csci566/demos/chair_ingolf/Sawyer_chair_ingolf_0650_0 --num_connects 1 --run_prefix p0 --gpu 0 --wandb False --wandb_entity amanbans --wandb_project policy-chaining --log_root_dir /media/aman-anmol/Aman/csci566/logs
#mpirun -np 16 python -m run --algo gail --furniture_name chair_ingolf_0650 --demo_path demos/chair_ingolf/Sawyer_chair_ingolf_0650_1 --num_connects 1 --preassembled 0 --run_prefix p1 --load_init_states log/table_lack_0825.gail.p0.123/success_00024576000.pkl
#mpirun -np 16 python -m run --algo gail --furniture_name chair_ingolf_0650 --demo_path demos/chair_ingolf/Sawyer_chair_ingolf_0650_2 --num_connects 1 --preassembled 0,1 --run_prefix p2 --load_init_states log/table_lack_0825.gail.p1.123/success_00030310400.pkl
#mpirun -np 16 python -m run --algo gail --furniture_name chair_ingolf_0650 --demo_path demos/chair_ingolf/Sawyer_chair_ingolf_0650_3 --num_connects 1 --preassembled 0,1,2 --run_prefix p3 --load_init_states log/table_lack_0825.gail.p2.123/success_00027852800.pkl

conda deactivate
