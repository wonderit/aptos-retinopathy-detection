# Kaggle APTOS 2019 Blindness Detection Solution


### Requirements

`pip3 install --user -r requirements.txt`


### Input data:
* [Diabethic Rethinopathy Detection (2015)](https://www.kaggle.com/c/diabetic-retinopathy-detection/data),
both train and test subsets
* [APTOS 2019 Blindness Detection (2019)](https://www.kaggle.com/c/aptos2019-blindness-detection/data),
 dublicate images from train subset are removed, 
 final label file: [input/trainLabels19_unique.csv](input/trainLabels19_unique.csv), 
 dublicate removal script: [preprocessing/2019_train_find_dublicates.py](preprocessing/2019_train_find_dublicates.py).
* [Messidor-1](http://www.adcis.net/en/third-party/messidor/), 
[[0-4] adjusted labels](https://www.kaggle.com/google-brain/messidor2-dr-grades), 
final label file: [input/messidor1_labels_adjudicated.csv](input/messidor1_labels_adjudicated.csv).

For 2015 and 2019 data, 
[this dataset](https://www.kaggle.com/benjaminwarner/resized-2015-2019-blindness-detection-images) 
with resized images was used. Images from Messidor dataset 
were also resized and converted to jpeg format using 
[preprocessing/messidor_tif_2_jpg.py](preprocessing/messidor_tif_2_jpg.py).

###  Train
`python3 train_efficientnet.py -sid N`

where N is Cross-Validation fold id, i.e. [0..4]

### Evaluate

`python3 eval_ensemble.py`