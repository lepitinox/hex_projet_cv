# hex_projet_cv

## Description
This is a educational project for the Hexagone school.
The goal si to use Computer Vision techniques to detect and categorize emotions.
Used Dataset is EMOTIC dataset.

## DataExploration
### FS structure
``` 
├───data
│   ├───emotic_test.csv
│   ├───emotic_train.csv
│   ├───test
│   │   └───data_source ...
│   │      └───images
│   │           └───<dataset_dependent_unique_id>.jpg
│   └───train
│       └───data_source ...
│          └───images
│               └───<dataset_dependent_unique_id>.jpg
```

### DataExploration
Training contain 12662 rows and 8 columns. With no missing values
of name : 'path, basename, extension, filename, last_modified, size, box, label'
```
path             /Archive/mscoco/images/COCO_val2014_0000005622...
basename                                 COCO_val2014_000000562243
extension                                                      jpg
filename                             COCO_val2014_000000562243.jpg
last_modified                             2024-01-02T23:47:19.768Z
size                                                         32778
box                                              [ 86  58 564 628]
label                                                Disconnection
```
Files size of grouped around 50Ko but with outliers up to 30Mo.
Files are in jpg format.
Box is a list of 4 int, representing the bounding box of the face.
Box data uses different format and will need dataprep.
Label is a string, representing the emotion.
There are 26 different emotions.
Dataset is unbalanced, with 3 emotions representing 65% of the dataset.
### Images
Some images have multiple faces, some have no face.