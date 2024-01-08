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
- Files size of grouped around 50Ko but with outliers up to 30Mo.
- Files are in jpg format.
- Box is a list of 4 int, representing the bounding box of the face.
- Box data uses different format and will need dataprep.
- Label is a string, representing the emotion.
- There are 26 different emotions.
- Dataset is unbalanced, with 3 emotions representing 65% of the dataset.
### Images
- Some images have multiple faces, some have no face.
- some images are in color, some are in black and white.

## Code Structure

Code base is split in 3 main parts:
- datamanagement: for data loading and preprocessing
- optimization: for data augmentation and reduction of computation time
- ai_pipeline: for model training and evaluation

## DataManagement
Datamanager will read the data and provide a data interface for the rest of the code.
### DataLoading
Data is loaded from csv files

### DataPreprocessing
Create new features from existing ones columns and clean data.

### DataAugmentation
Configurate the data augmentation pipeline.

## Optimization
Optimizers will plug in the datamanager
reading a dataframe and returning a dataframe.

### DataReduction
Configurate the data reduction pipeline.
### DataSpecificOptimization
Configurate the data specific optimization pipeline like face detection and dimension reduction.

## AI_Pipeline
Piplelines are scripts that will call the datamanager and optimizers to train and evaluate the model.
### ModelTraining 
Configurate the model training pipeline.

### ModelEvaluation
Configurate the model evaluation pipeline.