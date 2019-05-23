# multi-view-pose-estimation
This is the code for the paper ??? <br />

## Dependencies
Python 2.7 <br />
Pytorch v0.4.0 or Pytorch v0.1.12 <br />

## First of all
1- Watch our video: ??? <br />
2- Clone this repository and get the dataset. We provide the Human3.6M dataset in ground truth 3d points, camera rotation matrix, and estimated 2d points from Stacked Hourglass network alonge with corresponding confidence extracted from heatmaps.<br />
3- Edit genlist4view.py to set path-to-data and generate training/validation lists based on the data location.
```
git clone https://github.com/rmehrizi/multi-view-pose-estimation.git
cd multi-view-pose-estimation
wget https://www.googledrive/....
unzip dataset.zip
rm dataset.zip
python genlist4view.py
```

## A quick demo
1- For a quick demo, edit train4view.py to set path-to-data and change "demo_mode" to "True" and visualize the results on randomly selested images <br />
```
cd multi-view-pose-estimation
python train4view.py
```

## Testing
1- Edit train4view.py to set path-to-data and change "validation_mode" to "True" and check tracking results in multi-view-pose-estimation/result/. <br />
```
cd multi-view-pose-estimation
python train4view.py
```

## Training 
Training is performed in two steps: <br />
###### First Step (single-view):
1- Edit genlist.py to set path-to-data and generate training/validation lists for single view images. 
2- Edit train.py to set path-to-data and run it to train the model (2 blocks) and save the best model in exp/single-view <br />
```
cd multi-view-pose-estimation
python train.py
```
###### Second Step (multi-view):
1- Edit options/train-options to set "lr" to 0.0001 and "nEpochs" to 5 . <br />
2- Edit train4view.py to set path-to-data and change both "demo_mode" and "validation_mode" to "False".
3- Run train4view.py to train the model (4 blocks) and save the best model in exp/multi-view <br />
```
cd multi-view-pose-estimation
python train4view.py
```

## Citation
If you find this code useful in your research, please consider citing our work: ???

