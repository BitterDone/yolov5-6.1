0943 11/7/2025

sudo apt-get remove -y --purge python3
sudo apt-get autoremove -y

curl -fsSL https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init - bash)"' >> ~/.profile

# ? Not sure why this is here or what it does
exec "$SHELL"

sudo apt update; sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

pyenv install 3.9.0 
pyenv global 3.9.0

git clone https://github.com/BitterDone/yolov5-6.1.git
cd yolov5-6_1
pip install -r requirements.txt

pip uninstall torch -y
pip install torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

pip install numpy==1.24.4 --force-reinstall

# pip install numpy==1.24.4

# ERROR: After October 2020 you may experience errors when installing or updating packages. This is because pip will change the way that it resolves dependency conflicts.
# We recommend you use --use-feature=2020-resolver to test your packages with the new resolver before it becomes the default.
# opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you'll have numpy 1.24.4 which is incompatible.

pip install opencv-python==4.5.5.64
pip install pillow==9.5.0 --force-reinstall

# python train.py --img 640 --batch 16 --epochs 100 --data railroad-cars/data.yaml --weights weights/yolov5n-v6_1.pt --name railroad_detector

# sh backup-model-vm.sh
# sudo find / -type f -name "model_export.*" 2>/dev/null

# cd c:\r\me\monitoring-trains
# scp -i C:\r\me\monitoring-trains\monitoring-trains-key-openssh root@167.99.182.230:/root/model_export.zip .
