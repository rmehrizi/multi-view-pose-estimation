# multi-view-pose-estimation

Clone this repository and get the dataset. We provide the Human3.6M dataset in ground truth 3d points, camera rotation matrix, and estimated 2d points from Stacked Hourglass network alonge with corresponding confidence extracted from heatmaps.<br />
```
git clone https://github.com/rmehrizi/multi-view-pose-estimation.git
cd PoseEstomator
wget https://www.googledrive/....
unzip dataset.zip
rm dataset.zip
```

## A quick demo
1- For a quick demo, edit train4view.py to set path/to/caffe/python/ and change demo_mode to "True". <br />
2- Download the pre-trained model and visualize the results on randomly selested images. <br />
```
cd PoseEstomator
wget https://www.googledrive/....
python train4view.py
```

## Testing
1- Edit train4view.py to set path/to/caffe/python/ and change validation_mode to "True". <br />
2- Download the pre-trained model and Check tracking results in PoseEstimator/result/. <br />
```
cd PoseEstomator
wget https://www.googledrive/....
python train4view.py
```

##Training 
Training is performed in two steps:
###First Step (single-view):
1- Edit options/base-options to set "  " to " " . <br />
2- Edit options/train-options to set " lr " to " " . <br />
3- Run train.py to train the model (2 blocks) and save the best model in exp/single-view
```
cd PoseEstomator
python train.py
```
###Second Step (single-view):
1- Edit options/base-options to set "  " to " " . <br />
2- Edit options/train-options to set " lr " to " " . <br />
3- Run train4view.py to train the model (4 blocks) and save the best model in exp/multi-view
```
cd PoseEstomator
python train4view.py
```
