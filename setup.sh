# set up python 3.8
# good resource: https://www.itsupportwale.com/blog/how-to-upgrade-to-python-3-11-on-ubuntu-20-04-and-22-04-lts/
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.8
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

apt-get install python3-pip
apt-get install rsync
sudo apt-get install vim

git clone https://github.com/jonathanlin0/research_transfer.git

pip3 install git+https://bitbucket.org/pypy/numpy.git

pip3 install -r requirements.txt

# mkdir tmp
# cd tmp
# wget https://repo.anaconda.com/archive/Anaconda3-2023.07-0-Linux-x86_64.sh
# bash Anaconda3-2023.07-0-Linux-x86_64.sh
# source .bashrc

# conda create --name j_research python=3.10 pip

# # conda install --name j_research package

# # -force-reinstall: reinstall even if it is already installed
# # -y: yes, do not ask for confirmation
# # -c: additional channels
# conda install --name j_research -force-reinstall -y -c conda-forge --file requirements.txt

# conda create --name j_research -c conda-forge scikit-learn
# conda activate sklearn-env