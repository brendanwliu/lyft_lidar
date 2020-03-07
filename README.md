# lyft_lidar
An implementation of a UNet to solve a 3D lidar image problem.

## The Data
The data is structured like this:

    scene - Consists of 25-45 seconds of a car's journey in a given environment. Each scence is composed of many samples.
    sample - A snapshot of a scence at a particular instance in time. Each sample is annoted with the objects present.
    sample_data - Contains the data collected from a particular sensor on the car.
    sample_annotation - An annotated instance of an object within our interest.
    instance - An enumeration of all object instance we observed.
    category - Class of object categories (e.g. vehicle, human).
    attribute - Property of an instance that can change while the category remains the same.
    visibility - (currently not used)
    sensor - A specific sensor type.
    calibrated sensor - Definition of a particular sensor as calibrated on a particular vehicle.
    ego_pose - Ego vehicle poses at a particular timestamp.
    log - Log information from which the data was extracted.
    map - Map data that is stored as binary semantic masks from a top-down view.
    
Each frame in the data set comes in two forms of data: LiDAR and image data. Image data is realatively easy to deal with, as they come in RGB/jpeg format. These can be fed into a Convolutional Neural Network (CNN) as a 3D tensor (channels, width, height).

Now time to tackle the biggest problem of the project: what is LiDAR and how can I train a CNN on this data?

LiDAR (Light Detection and Ranging) is a image capture method that is used to generate 3D representations of the surroundings. A laser is emitted from the camera, and reflected off the environment back to the camera sensor. The time required for the light to reflect back to the camera sensor is calculated. LiDAR camera works like this:

![gif of lidar camera](https://i.imgur.com/Frl3hgk.gif)

This is extremly powerful because LiDAR can give the depth information of an object when combined from the 2D camera information, similar to how humans can see using inputs from both eyes.

## EDA



## 
