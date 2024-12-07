# GoogleEarthEngineTask
Several Tasks about using Google Earth Engine.

## Tif_ML

This is folder for using Google Earth Engine to generate training and testing data for Machine Learning.

Basedon using the Ncfi and sentinel

### Processing

- Collect the tif file from Google Earth Engine
- Generate the target tif from the cliped shape and cross with the nicfi tif range by using the NASA fire points data shape
    - Target tif generate Overlap , and buffering the point to circle
- Resampling the shape between the nicfi and sentinel, make the data be same resolution
    - Downscalling the nicfi to sentinel resolution

- Minimize the data scall. only choose 100 dp for 3 month, 300 nicfi image ,1000 sentinel image ,300 target

- Data ready for training:
    - option1: Tabular data ,merge the all value from 4x nicfi, 26 sentinel x 3 image/month 
    - Tif , cnn model
    -
