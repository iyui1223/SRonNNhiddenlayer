# final_project
for final project for MPhil DIS: Symbolic distillization.



# Access and env-setup at CSD3

# Anaconda environment for project can be loaded with 
conda activate final; module load gcc/11.3.0 # needs latest gcc for torch-scatter, torch-sparce

# using computation node
Userguide for CSD3, regarding interactive login.
https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html#software

Check the status of nodes -- 
$ sinfo

Check which nodes do I have access to --
$ sacctmgr show user $USER format=User,Account,Partition

Run interactive job at, for example, ampere with YOURPROJECT-GPU
$ sintr -t 4:0:0 --exclusive -A YOURPROJECT-GPU -p ampere
