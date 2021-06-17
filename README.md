# lyft_lidar
An implementation of a UNet to solve a 3D lidar image problem. This project is part of a project with the UCSB Data Science Club and is at this point 'complete'.

## Final Comments Post-Project
This was my first foray into utilizing CNNs in a real world problem, drawing on examples already in the deeplearning community. It was an interesting experience learning and understanding image segmentation and its implication on self driving cars. By no means is this a comprehensive experiment. I just wanted to acknowledge the help of the UCSB Data Science Club and the Statistics faculty for helping me gain interest in the world of computational statistics. This repository is now archived.

## Abstract
Self-driving cars are becoming less and less of a sci-fi dream every year. Each year over 42,000 people die of car accidents and over 2.7 million are injured. Being able to create a autonomous car that has never gets tired, makes objective decisions based on millions of hours of training data will improve safety in the long run. I trained a covolutional neural network (CNN) to segment birds eye view LiDAR images. Using the lyft level5 API I was able to get my annotated traning and validation datasets from lyft cars being driven around palo alto. My results are not fantastic;and, on the Kaggle leaderboard, I was able to score an abysmal 0.01 on the private held out data set, with the winner scoring a 0.216. However, the experiment still proves useful, because we can explore the segmentation results based on LiDAR data alone and compare it to the top scorers, which use a combination of image and LiDAR data.

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

Let's take a look at the train dataframe to see what metadata we're given with each file.

![train dataframe](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/train_df.png)

The center_x(center_y)(center_z) values are the width, depth, and height values from the center of the camera, we can explore these with seaborn distribution plots to see where a majority of objects lie with respect to the camera.

![centerxy distn](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/center_xy_distn.png)

We can see that center_x (orange) is more evenly distributed than center_y (purple). This is because the camera can capture/sense objects on the left and right really well, but not the depth of the object. The length of the road in front is much longer than the width of the road. This is where I believe that the LiDAR data will come in very handy. The birds eye view from the top of the car can be extremely useful in helping these CNN's perceive depth in the environment.

Next we look at the distribution of center_z:

![centerz distn](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/center_z_distn.png)

We can see that the distribution of center_z has an extreme rightward skew and is clustered around -20. The variation of center_z is significantly smaller than that of center_x and center_y. This is probably because most objects are very close to the flat plane of the road, and therefore, there is no great variation in the height of the objects above (or below) the camera. 

Also, most z coordinates are negative because the camera is attached at the top of the car. So, most of the times, the camera has to "look down" to see the objects. Therefore, the height or z-coordinate of the objects relative to the camera are generally negative. We can see that there are two large spikes in center_z, which are probably the heights of the two most common classes, pedestrians and cars.

Next let's look at the distributions of classes in our dataset:

![distn classes](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/distn_of_classes.png)

As expected the car is the most represented class.

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

With the accompanying camera data:

![camera vis sample 1](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/camera_vis_sample_1.png)

## LiDAR Representation

Here comes the difficult part of the project. How can I represent LiDAR data and what is the most effective representation of LiDAR data. Well we already know that cameras don't preceive depth. Much of the literature around LiDAR data uses the birds eye view representation. How can we transform the 3D point cloud data to a 2D image representation? By using the voxel.

'A voxel represents a single sample, or data point, on a regularly spaced, three-dimensional grid. This data point can consist of a single piece of data, such as an opacity, or multiple pieces of data, such as a color in addition to opacity. A voxel represents only a single point on this grid, not a volume; the space between each voxel is not represented in a voxel-based dataset. Depending on the type of data and the intended use for the dataset, this missing information may be reconstructed and/or approximated, e.g. via interpolation.'

The value of a voxel may represent various properties. Here each voxel stores the height of the lidar data separated into three bins, which are visualized like RGB channels in an image in a birds eye view(bev).

As input for our network we voxelize the LIDAR points. That means that we go from a list of coordinates of points, to a X by Y by Z space. Voxels can contain multiple scalar values, essentially vector (tensor) data; in the case of ultrasound scans with B-mode and Doppler data, density, and volumetric flow rate are captured as separate channels of data relating to the same voxel positions.

Here it is used to represent the height of our lidar data for our CNN to train on.

Here I use these hyperparameters:

    voxel_size = (0.4,0.4,1.5)
    z_offset = -2.0
    bev_shape = (336, 336, 3)

I found in my training that a voxel that is longer in height than in width and longer worked better than prefectly square voxels. This is because we only have 3 channels to work with (RGB) and want to cover as many objects as possible, so my voxels are super coarse. If we were to have voxel sizes of (1,1,1), it would be like the videogame minecraft. We need a z_offset because again, the camera is mounted to the top of the car and the car is about 2.0m tall. Thanks to gzuidhof for the LiDAR to voxel transformation functions.

![lidar top sample](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/lidar_top_sample_1.png)

This is an example of the lidar_top raw data visualized using lyft SDK's render_sample_data function. Below is the data after being transformed into voxel space.

![voxel transform sample](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/sample_processed_data.png)

Above is an example of what the input for our network will look like. It's a top-down projection of the world around the car (the car faces to the right in the image). The height of the lidar points are separated into three bins, which visualized like this these are the RGB channels of the image.

Now that we've created our inputs and targets, let's setup our model and train.

## Limitations - Technical and otherwise

I want to take the time to explore the limitations of my experiment. Any batch size above 8 was too large to fit in memory, even downscaling the images in half. I also did not augment the data, as explained in the original UNet paper, was quintessential to the success of the architechture. The UNet also takes all samples as IID, which it is not. We know the data is temporal but do not take this into account. I was originally using a GCP instance with free education credit to do the computataion, however, I spent all the credit and did not have the money to continue. 

## Training a UNet to Segment BEV Images

First we explore the UNet model originally proposed by [Ronneberger et al](https://arxiv.org/pdf/1505.04597.pdf). 

    "The main idea in the Unet is to supplement a usual contracting network by successive layers, where pooling operators are
    replaced by upsampling operators.Hence, these layers increase the resolution of the output. In order to localize,
    highresolution features from the contracting path are combined with the upsampled output. A successive convolution layer
    can then learn to assemble a more preciseoutput based on this information." (Ronneberger et al., 2)

This yields the u-shaped architechture:

![UNet Model](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/UNetModel.png)

However, as stated in the original paper, this network is supposed to be able to be trained on very few data points in the training set, which is beneficial for bio-medical applications due to lack of traning data. This CNN can be trained end to end to label at the pixel level, with each pixel being given a class label. The energy function is computed by a pixel-wise softmax overt the final feature map combined with the cross entropy loss function. We use the softmax activation function to predict the class of each pixel, interpretable as a probability:

![softmax](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/softmax.png)

Our softmax is maximized for a certian class k if pk(x) = 1 while all other pk(x) = 0. The crossentropy penalizes, at each pixel, the deviation of our softmax from 1:

![crossentropy](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/crossentropy.png)

There's still one more problem that needs to be addressed. A vast majority of the predicted objects will be background noise that are not the classes that we are interested in 

    ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]

Therefore we need a class for all these "background" objects, and we weight the loss for the "background" class to be lower than all other classes to account for the large class imbalance (with cars being vastly over represented).

In the following, there are four different visualizations stacked on top of eachother:

    1. The top images have two color channels: red for predictions, green for targets. Note that red+green=yellow. In other words:

        Black: True Negative
        Green: False Negative
        Yellow: True Positive
        Red: False Positive

    2. The input image
    3. The input image blended together with targets+predictions
    4. The predictions thresholded at 0.5 probability.


I trained the network on the lyft level5 data for 25 epochs and visualized the results of the first two samples at each epoch. At epoch 1 we can see that obviously the network has not trained for long enough and is really bad at segmenting objects with a mean error of 0.20.

![first epoch](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/UNet_epoch_1_1.png)

The first image is the error between the predicted values and the ground truth, with red being false positives. We can also see that there seems to be no fourth image, thats because our net thinks everything is an object, causing the rediculous false positive rate.

After 25 epochs, we can see that the model has converged more and the mean error is a lot less of 0.02:

![25th epoch](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/UNet_epoch_25.png)

We can see that the CNN can segment the image better, however we need to compute the mean average percent (mAP). You can read a succinct article about mAP ![here](https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52#1a59). At a IoU of 0.5, I was able to get an mAP of 0.20196 on average over all classes. I looked at the image with the highest validation loss:

![high val loss](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/UNet_highest_val_loss.png)

We can see for the majority of cars we can see that the network can see them, however, for the small cluster of green bicycles the UNet doesn't see it at all. There's also a problem with a ton of false positives, but this could be because the model has not converged yet. Though the mean error decreased per epoch, I don't believe that the model had converged yet.

We can see that the network struggles to detect smaller objects and I plotted a specific example with a pedestrian in the environment. 

![small obj](https://github.com/brendanwliu/lyft_lidar/blob/master/aux_files/images/drawing_bounding_boxes.png)

The small pedestrian we can see on the very right of the image in the target plot was not a predicted object by the UNet.I'm still trying to perform more analytics, like breaking down the mAP by class. I will update the readme once I'm done.

Problems with the model:

    1. Model performs badly on uncommon classes. This is what I assume is dragging down the overall mAP of our model
    2. The boxes are super bad, because of our super coarse voxels
    3. I didn't have money to train the model for longer so it can converge (it cost ~$35-$40 per training of my model)
    4. We say that all objects can be broken down into three different height buckets, but this is inherently false as we can have cars that span more than one height 'bucket' and all objects are not at the height of the vehicle gathering the data.
    5. We only use LiDAR data, with most state of the art CNN's using LiDAR in conjunction with camera data.

## Future Work

    1. Data augmentation - see what the effects of translating and deforming traning samples
    2. Combining CNN's that use camera data and lidar data 
    3. Exploring pseudo-bev mappings that only use camera data to make synthetic bev images
    
Thank you for taking the time to read the learnings of a college senior! 
