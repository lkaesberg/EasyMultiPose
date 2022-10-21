# EasyMultiPose

## Installation
```
git clone --recurse-submodules https://github.com/lkaesberg/EasyMultiPose.git
cd EasyMultiPose
conda env create -n easymultipose --file environment.yaml
conda activate easymultipose
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar

cd cosypose
python cosypose/setup.py install
```

## Train
```
conda activate easymultipose

# Change config file
python easymultipose/train/train_cosypose_detector.py


# Change config file
python easymultipose/train/train_cosypose.py
```
