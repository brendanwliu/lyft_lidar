# lyft_lidar
An implementation of a UNet to solve a 3D lidar image problem.

## Abstract
Self-driving cars are becoming less and less of a sci-fi dream every year. Each year over 42,000 people die of car accidents and over 2.7 million are injured. Being able to create a autonomous car that has never gets tired, makes objective decisions based on millions of hours of training data will improve safety in the long run. I trained a covolutional neural network (CNN) to segment birds eye view LiDAR images. Using the lyft level5 API I was able to get my annotated traning and validation datasets. My results are not fantastic, on the Kaggle leaderboard, I was able to score an abysmal 0.01 on the 70% private held out data set, with the winner scoring a 0.216. However, the experiment still proves useful, because we can explore the segmentation results based on LiDAR data alone and compare it to the top scorers, which use a combination of image and LiDAR data.

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

### Scenes

A scene is a 25-45s long sequence of consecutive frames extracted from a log. A frame (also called a sample) is a collection of sensor outputs (images, lidar points) at a given timestamp:

    scene {
       "token":                   <str> -- Unique record identifier.
       "name":                    <str> -- Short string identifier.
       "description":             <str> -- Longer description of the scene.
       "log_token":               <str> -- Foreign key. Points to log from where the data was extracted.
       "nbr_samples":             <int> -- Number of samples in this scene.
       "first_sample_token":      <str> -- Foreign key. Points to the first sample in scene.
       "last_sample_token":       <str> -- Foreign key. Points to the last sample in scene.
    }
    
We can have a look at what the raw LiDAR data looks like and what we're dealing with. By using the built in function in lyft's SDK we can see a more effective 3D visualization:

![lidar3d visualization sample 1](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/lidar3d_vis_sample_1.png)

Side by side with the camera data:


## 
