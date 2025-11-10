# 0943 11/7/2025
# TOKEN=token
# git clone https://BitterDone:$TOKEN@github.com/BitterDone/yolov5-6.1.git
# cd yolov5-6_1

curl -fsSL https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init - bash)"' >> ~/.bash_profile

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init - bash)"' >> ~/.profile

# Reloads the shell with new env vars
exec "$SHELL"

# # Try these next time
# source ~/.bashrc
# source ~/.bash_profile
# source ~/.profile

# Install modules for pyenv install 3.9.0
sudo apt update && sudo apt install -y \
  make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev \
  libncursesw5-dev xz-utils tk-dev \
  libffi-dev liblzma-dev python3-openssl git curl

# pyenv install 3.9.0 
# pyenv global 3.9.0
# pyenv virtualenv 3.9.0 yolov5-env
# # Error about not having env vars present in the shell?
# pyenv activate yolov5-env

# pip install -r requirements.txt

# pip uninstall torch torchvision torchaudio -y
# pip cache purge
# pip install networkx==3.1
# pip install torch==2.2.0+cu118 torchvision==0.17.1+cu118 torchaudio==2.3.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118


# python train.py --img 640 --batch 16 --epochs 100 --data railroad-cars/data.yaml --name railroad_detector
# python train.py --img 640 --batch 16 --epochs 300 --data railroad-cars/data.yaml --name railroad_detector
# python train.py --img 640 --batch 16 --epochs 600 --data railroad-cars/data.yaml --name railroad_detector
# python train.py --img 640 --batch 16 --epochs 100 --data railroad-cars/data.yaml --name railroad_detector_aug --image-weights  --hyp data/hyps/hyp.scratch-custom.yaml
# python train.py --img 640 --batch 16 --epochs 300 --data railroad-cars/data.yaml --name railroad_detector_aug --image-weights  --hyp data/hyps/hyp.scratch-custom.yaml
# python train.py --img 640 --batch 16 --epochs 600 --data railroad-cars/data.yaml --name railroad_detector_aug --image-weights  --hyp data/hyps/hyp.scratch-custom.yaml

# python val.py --weights runs/train/railroad_detector/weights/best.pt --data railroad-cars/data.yaml --img 640 --batch 16

# git add -f weights/ opt.yaml hyp.yaml results.png results.csv

# sh backup-model-vm.sh
# sudo find / -type f -name "model_export.*" 2>/dev/null

# cd c:\r\me\monitoring-trains
# scp -i C:\r\me\monitoring-trains\monitoring-trains-key-openssh root@167.99.182.230:/root/model_export.zip .
