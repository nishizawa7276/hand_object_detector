conda create --name handobj_new python=3.8
conda activate handobj_new
conda install -c anaconda cudatoolkit=11.3 cudnn
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1  cudatoolkit=11.3 cudnn -c pytorch
pip install -r requirements.txt
conda install -c conda-forge cudatoolkit=11.3 cudnn

export LD_LIBRARY_PATH=/home/knishizawa/.conda/envs/handobj_new/lib:$LD_LIBRARY_PATH
export PATH=/home/knishizawa/.conda/envs/handobj_new/bin:$PATH
export LD_LIBRARY_PATH=/home/knishizawa/.conda/envs/handobj_new/lib:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/home/knishizawa/.conda/envs/handobj_new/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 python demo.py --cuda --checkepoch=8 --checkpoint=89999

faster_rcnn_1_8_89999



wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-3
