# DL

To train the model,
Go to hpc account and cd to home/<NETID>
The environment is miniconda (details could be seen in https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda)
Note that in the installation part, 
````
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

pip3 install jupyter jupyterhub pandas matplotlib scipy scikit-learn scikit-image Pillow tqdm numpy
````
Then. get the dataset from /scratch/
````
cp -rp /scratch/DL23SP/dataset_v2.sqsh /scratch/USERID/
unsquashfs dataset_v2.sqsh
````
The file will be in home/NETID/squashfs-root/dataset

Also, get the hidden dataset by 
````
pip install gdown
gdown https://drive.google.com/uc?id=12Ma4Ssm6uFzEpvfdRofM5uhbiWxwpZke
unzip hidden_set_for_leaderboard_1.zip
````
Then, 
````
git clone  https://github.com/MAOQIZHANG/DL.git
````
Then move all the sbatch file in the folder to current folder and submit it using sbatch
````
mkdir out
cp -r DL/*.sbatch .
````
**This step is important since all path are based on root dir. 
First, train the model and wait for it to generate a model.pth. 
````
mkdir model_output
sbatch run-test64.sbatch
````
To skip the training time which is 12 hours, 
use 
````
cp DL/model/best_conv_lstm_model_b64_l3_e50.pth model_output/.
````
Then run predict on it
````
sbatch run-predict64.sbatch
````
The temporary outcome will be saved in 
Then generate mask. 
````
mkdir final
sbatch run-mask64.sbatch
````
The final outcome of mask will be in final. 

````
$ls final
final_64.npy
````