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
For a quick demo, you can download the pre-trained model and visualize the results on randomly selested images. <br />
```
cd PoseEstomator
wget https://www.googledrive/....
python train4view.py
```

## Validation


