#**Behavioral Cloning** 



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/centre.png "Centre Lane Driving"
[image3]: ./images/recovery1.png "Recovery Image"
[image4]: ./images/recovery2.png "Recovery Image"
[image5]: ./images/center_2017_05_02_19_44_01_107.png "Normal Image"
[image6]: ./images/center_2017_05_02_19_44_01_107_flipped.png "FLipped Image"
[image7]: ./images/center_2017_05_02_18_33_51_889.png "Normal Image"
[image8]: ./images/center_2017_05_02_18_33_51_889_cropped.png "Cropped Image"
[image9]: ./images/history.png "Training History"

#### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation. 

---
###Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* predict.py for sanity check of the model (This is not rquired for training or driving)
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 the video of car driving around the track with the final model

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

For the final model I used convolutional neural network used by Bojarski et al. at NVIDIA[Bojarski et al. at NVIDIA](https://arxiv.org/pdf/1604.07316.pdf) .



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py see function nvidia()). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py main function).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving slowly around the curves.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started by using LeNet architecture (model.py function lebet()) to verify the python code framework is working correctly.

Next I selected the NVIDIA CNN by Bojarski et al. since it has provesn to be successfull in navigating real life scenarios.

Modifications to the model.
- To combat the overfitting: I added dropout layers after each layer with a dropout fraction of 20%.
- To introduce non-linearities: For activation functions I used Exponential Linear Units (ELU) to introduce non-linearities as this allows negative values (in contrast to RELU), thereby allowing the mean value of activations to be close to zero. [This improves performance of optimizers](https://arxiv.org/abs/1511.07289v1) This improves performance of optimizers.

Furthermore I used a small dataset to verify the python code. The small dataset contains several images with steering angles approximately -1,0 and 1. I used predict.py to verify that the predictions of the model are as expected when applied to the same dataset as a sanity check.

```sh
python predict.py model.h5
```

#### 2. Final Model Architecture

The final model architecture (model.py function nvidia() ) consisted of a convolution neural network with the following layers and layer sizes .

| No.|Layer         		|     Description	        					| 
|:--:|:---------------------:|:---------------------------------------------:| 
|1| Input         		| 32x32x3 Normalized image   							| 
|| Convolution 5x5     	| 2x2 stride, filters 24, valid padding 	|
|| ELU					|												|
|| Dropout	      	| 50% dropout rate 				|
|2| Convolution 5x5     	| 2x2 stride, filters 36,valid padding 	|
|| ELU					|												|
|| Dropout	      	| 20% dropout rate 				|
|3| Convolution 5x5     	| 2x2 stride, filters 48, valid padding 	|
|| ELU					|												|
|| Dropout	      	| 20% dropout rate 				|
|4| Convolution 3x3     	| 2x2 stride, filters 64, valid padding 	|
|| ELU					|												|
|| Dropout	      	| 20% dropout rate 				|
|5| Convolution 3x3     	| 2x2 stride, filters 64, valid padding 	|
|| ELU					|												|
|| Dropout	      	| 20% dropout rate 				|
|| Flatten	      	| 			|
|6| Fully connected		| input 1164, output 100									|
|| ELU					|												|
|| Dropout	      	| 20% dropout rate 				|
|7| Fully connected		| input 100, output 50									|
|| ELU					|												|
|| Dropout	      	| 20% dropout rate 				|
|8| Fully connected		| input 50, output 10									|
|| ELU					|												|
|| Dropout	      	| 20% dropout rate 				|
|9| Fully connected		| input 10, output 1									|
|| ELU					|												|
|| Dropout	      	| 20% dropout rate 				|

Here is a visualization of the architecture generated using keras utils.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

##### Data is king!

It turned out that one of the biggest factors of success was the quality of data. In order to improve the dataset I did the following.

- Started with the dataset provided by udacity as it gave good results (it did not make it all the way) and added to it.
- Added a lap of centre lane driving.

![alt text][image2]

- Added a lap of recovery driving from either side of the road. I only recorded data when the car is moving towards the centre to prevent the NN learning to go off the road.

![alt text][image3]
![alt text][image4]

- I used a joystick to collect data, since it provided musch smoother steering angles

##### Augmentation

I copied the entire dataset doubling the number of data points. This second copy was modified in the following way.

- 40%. Flipped the image horizontally and negated the steering angle in order to counteract the left turning bias of the track.
![alt text][image5]      ![alt text][image6]
- 30% each. Randomly cropped out left or right half of the image. This was done to force the car to learn to follow the road using one curb. (in a form of directed dropout). I did this to prevent the car from going offroad where there is no curb on one side. (When the model was trained with dataset provided the car went of the road at the curve after the bridge where there is no curb on the right hand side. This was rectified by this method of augmentation)
![alt text][image7]  ![alt text][image8]


##### Preprocessing

After the collection process, I had 19868 number of data pointsc(31788 after augmentation). I then preprocessed this data by cropping out the top and bottom of the image to remove the hood of the car (which is unchanging and therefore not contributing learning) and the sky and background (which contains information not required and distracting to learning to stay on the track). 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

##### Training
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. However, in general training loss was slightly higher than the validation loss. 

I used an adam optimizer. In order to use the optimal number of epochs I used a small learning rate of 0.0005 and and a large number of epochs (100) with early stopping enabled. This way keras will stop training once the loss seizes to improve.

![alt text][image9]
