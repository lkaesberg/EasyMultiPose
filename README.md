# EasyMultiPose

## Installation
```
git clone --recurse-submodules https://github.com/lkaesberg/EasyMultiPose.git
cd EasyMultiPose
conda env create -n easymultipose --file environment.yaml
conda activate easymultipose
wget https://owncloud.gwdg.de/index.php/s/qX9eXVaB6aOkACi/download -O VOCdevkit.tar
tar -xvf VOCdevkit.tar

cd cosypose
python setup.py install
```

## Train
```
conda activate easymultipose

# Change config file
python easymultipose/train/train_cosypose_detector.py


# Change config file
python easymultipose/train/train_cosypose.py
```
