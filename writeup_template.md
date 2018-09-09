# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writen_image/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_image/1.png "Traffic Sign 1"
[image5]: ./test_image/2.png "Traffic Sign 2"
[image6]: ./test_image/3.png "Traffic Sign 3"
[image7]: ./test_image/4.png "Traffic Sign 4"
[image8]: ./test_image/5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.
1、Files submitted, from everything I read a HTML file, notebook, and write up file is required. So this will meet requirements.
2、Dataset summary & visualization, the rubric is referring to explaining the size of the dataset and shape of the data therein. It also would like visual explorations.
3、Design & test model: which includes preprocessing, model architecture, training, and solution. This was basically given to us for the most part in the with LeNet. The preprocessing from what I have read is many common tasks especially in the paper referenced by the instructors. It outlined the types of data preprocessing and I have tried to implement as much I could time permitting. This includes changing from 3 color channels to 1 so grayscale and then also normalizing the image data so it is smaller numbers. I also did some random augmentation mostly slight edging of the image in random directions, or tilting and then adding those new images to the data set and redistributing it to the train and validation sets while leaving the test set alone. For training again this was basically given using the AdamOptimizer. It worked really well so I didn't change it from the last quiz before this project. The more important parts of the training in the instance I think is the epoch which was 27 and batch size was 158. I also used a learning rate of 0.00097 because it gave good results. Lastly in regards to the solution, or model design I used the default given to me except I did add two drops outs and adjusted the size of the layers to better represent the actual data since it is 32x32 and not 28x28. I also added another convolution.
4、Test model on new images, I found new images on the internet and tried to find images that were already classified out of the 43 classes. It wasn't difficult, but at first I did try images that were very difficult to classify and it didn't do that well. After I found images of signs that were severely damaged it identified the images fairly well. I did scale my images perfectly to 32x32 as well which is probably some what limiting to real scenarios. 
5、Finally I also show the probabilities reduced for softmax probabilities and also individual performance along with total performance for all.

You're reading it! and here is a link to my [project code](https://github.com/Jock-Wang/My-CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data 

I decided to generate additional data  

To add more data to the the data set, I used the following techniques 

Here is an example of an original image and an augmented image:

![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x8 	|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 24x24x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 12x12x16 				|
| Convolution 1x1		| 2x2 stride, valid padding, outputs 10x10x32 									|
| RELU					|												|
| Convolution 1x1		| 2x2 stride, valid padding, outputs 8x8x32 									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 4x4x32 				|
| Flatten	      	| Input = 4x4x32. Output = 512 				|
| Fully connected Layer 1	      	| Input = 512. Output = 120 				|
| RELU					|												|
| dropout	      	| 50% keep 				|
| Fully connected Layer 2	      	| Input = 120. Output = 84 				|
| RELU					|												|
| Fully connected Layer 3	      	| Input = 84. Output = 43 				|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a new net for the most part that was given, but I did add an additional convolution without a max pooling layer after it like in the udacity lesson. I used the AdamOptimizer with a learning rate of 0.001. The epochs used was 27 while the batch size was 128. Other important parameters I learned were important was the number and distribution of additional data generated. I played around with various different distributions of image class counts and it had a dramatic effect on the training set accuracy. It didn't really have much of an effect on the test set accuracy, or real world image accuracy.  When I finally stopped testing I got 0.971 accuracy on the test set.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.971
* test set accuracy of 0.946

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I used a very similar architecture to the paper offered by the instructors.
* What were some problems with the initial architecture?
The first issue was lack of data for some images and the last was lack of knowledge of all the parameters.
* How was the architecture adjusted and why was it adjusted? 
Past what was said in the previous question, I didn't alter much past adding a couple dropouts with a 50% probability.
* Which parameters were tuned? How were they adjusted and why?
Epoch, learning rate, batch size, and drop out probability were all parameters tuned along with the number of random modifications to generate more image data was tuned. For Epoch the main reason I tuned this was after I started to get better accuracy early on I lowered the number once I had confidence I could reach my accuracy goals. 
* What are some of the important design choices and why were they chosen? 
 I think I could go over this project for another week and keep on learning. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)     		| Speed limit (30km/h)   									| 
| Bumpy road     			| Bumpy road 										|
| Ahead only					| Ahead only											|
| Priority road	      		| No vehicles					 				|
| Go straight or left			| Go straight or left      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 




