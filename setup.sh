conda create --name j_research python=3.10 pip

# conda install --name j_research package

# -force-reinstall: reinstall even if it is already installed
# -y: yes, do not ask for confirmation
# -c: additional channels
conda install --name j_research -force-reinstall -y -c conda-forge --file requirements.txt

conda create --name j_research -c conda-forge scikit-learn
conda activate sklearn-env