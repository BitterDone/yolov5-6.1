@echo off
set REMOTE_USER=root
set REMOTE_HOST=167.99.182.230  # e.g., 137.184.XX.XX
set REMOTE_PATH=runs/train/railroad_detector7
set REMOTE_DATA_YAML=railroad-cars/data.yaml
set LOCAL_DEST=railroad_model_backup

REM Create local directory if it doesn't exist
if not exist %LOCAL_DEST% (
    mkdir %LOCAL_DEST%
)

echo 📥 Downloading best.pt
scp %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_BASE_PATH%/weights/best.pt %LOCAL_DEST%\

echo 📥 Downloading results.png
scp %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_BASE_PATH%/results.png %LOCAL_DEST%\

echo 📥 Downloading opt.yaml
scp %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_BASE_PATH%/opt.yaml %LOCAL_DEST%\

echo 📥 Downloading hyp.yaml
scp %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_BASE_PATH%/hyp.yaml %LOCAL_DEST%\

echo 📥 Downloading train_batch*.jpg
scp %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_BASE_PATH%/train_batch*.jpg %LOCAL_DEST%\

echo 📥 Downloading data.yaml
scp %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DATA_YAML% %LOCAL_DEST%\data.yaml

echo ✅ All files downloaded to %LOCAL_DEST%
pause
