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

# # Reloads the shell with new env vars
# exec "$SHELL"

# Try these next time
source ~/.bashrc
source ~/.bash_profile
source ~/.profile

# Install modules for pyenv install 3.9.0
sudo apt update && sudo apt install -y \
  make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev \
  libncursesw5-dev xz-utils tk-dev \
  libffi-dev liblzma-dev python3-openssl git curl

pyenv install 3.9.0 
pyenv global 3.9.0
pyenv virtualenv 3.9.0 yolov5-env
# Error about not having env vars present in the shell?
pyenv activate yolov5-env

pip install -r requirements.txt

pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install networkx==3.1
pip install torch==2.2.0+cu118 torchvision==0.17.1+cu118 torchaudio==2.3.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# python train.py --img 640 --batch 16 --epochs 100 --weights "" --cfg models/yolov5s.yaml --data railroad-cars/data.yaml --name railroad_detector
# python train.py --img 640 --batch 16 --epochs 300 --weights "" --cfg models/yolov5s.yaml --data railroad-cars/data.yaml --name railroad_detector
# python train.py --img 640 --batch 16 --epochs 600 --weights "" --cfg models/yolov5s.yaml --data railroad-cars/data.yaml --name railroad_detector
# python train.py --img 640 --batch 16 --epochs 100 --weights "" --cfg models/yolov5s.yaml --data railroad-cars/data.yaml --name railroad_detector_aug --image-weights  --hyp data/hyps/hyp.scratch-custom.yaml
# python train.py --img 640 --batch 16 --epochs 300 --weights "" --cfg models/yolov5s.yaml --data railroad-cars/data.yaml --name railroad_detector_aug --image-weights  --hyp data/hyps/hyp.scratch-custom.yaml
# python train.py --img 640 --batch 16 --epochs 600 --weights "" --cfg models/yolov5s.yaml --data railroad-cars/data.yaml --name railroad_detector_aug --image-weights  --hyp data/hyps/hyp.scratch-custom.yaml

# python val.py --weights runs/train/railroad_detector/weights/best.pt --data railroad-cars/data.yaml --img 640 --batch 16

# git add -f weights/ opt.yaml hyp.yaml results.png results.csv

# sh backup-model-vm.sh
# sudo find / -type f -name "model_export.*" 2>/dev/null

# cd c:\r\me\monitoring-trains
# scp -i C:\r\me\monitoring-trains\monitoring-trains-key-openssh root@167.99.182.230:/root/model_export.zip .


   # 13  python detect.py --weights runs/train/railroad_teacher/weights/best.pt --source railroad-cars/train/images --save-txt --save-conf --project pseudo_labels --name train
   # 14  python detect.py --weights runs/train/railroad_teacher/weights/best.pt --source railroad-cars/valid/images --save-txt --save-conf --project pseudo_labels --name valid
   # 20  cd pseudo_labels/train
   # 31  mkdir images && mv ./*.jpg ./images && cd ../valid
   # 40  mkdir images && mv ./*.jpg ./images
   # 44  cd ../train/
   # 45  git add -f labels/
   
# Step #3
   # 12  mv railroad-cars/train/labels railroad-cars/train/labels_orig
   # 16  cp -r pseudo_labels/train/labels railroad-cars/train/
   # 18  python train.py --img 640 --batch 32 --epochs 600 --data railroad-cars/data.yaml --cfg models/yolov5n.yaml --weights '' --name railroad_student
   # 20  python train.py --img 640 --batch 32 --epochs 600 --data railroad-cars/data.yaml --cfg models/yolov5n.yaml --weights '' --name railroad_student
   # 21  cd railroad-cars/train/labels

   # 34  for f in ./*.txt; do   awk '{if (NF==6) {print $1, $2, $3, $4, $5} else {print $0}}' "$f" > "${f}.tmp" && mv "${f}.tmp" "$f"; done
   # 35  awk '{print NF}' *.txt | sort | uniq -c
   
   # 38  python train.py --img 640 --batch 32 --epochs 600 --data railroad-cars/data.yaml --cfg models/yolov5n.yaml --weights '' --name railroad_student
   # 39  vi runs/train/railroad_student3/training-results.txt
   # 40  python val.py --weights runs/train/railroad_student3/weights/best.pt --data railroad-cars/data.yaml --img 640 --batch 16

# Step #4
   # 52  pip uninstall openvino openvino-dev -y
   # 53  pip install openvino==2022.3.1 openvino-dev==2022.3.1 tensorflow
   # 43  python export.py     --weights runs/train/railroad_student3/weights/best.pt     --include torchscript onnx openvino tflite

# Step #5
# python detect.py --weights runs/train/railroad_student3/weights/best.pt --source railroad-cars/train/images --conf-thres 0.5 --save-txt --save-conf
