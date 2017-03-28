#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the shape to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 4410
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

In the report.html file you will find a histogram showing the frequencies of the each sign class in the training data set.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

1. I "perturbed" the training set to add more variations.. Perturbing took the form a shifting the image up to 2 pixels in either the X or Y direction and rotating the image upto 15 degrees either clockwise or counter-clockwise.
2. All images were converted to grayscale.  I did this after the "Traffic Sign Recognition with Mutli-Scale Convolutional Networks" paper where they state they didn't use color.  This also makes sense as there are color blind drivers and they will need to recognize signs without relying on a color information
3. The data was normalized to values between .1 and .9

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The training and validation (and testing) set were already split in the downloaded ZIP file. So I left that as is, but did extend the number of training images by perturbing that data.  For more detail on perturbing, See #1.1  

My final training set had 69598 number of images. My validation set had 4410 images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grey scaled, normaized image 			| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flattening			| Outputs 400 (5 * 5 * 16)						|
| Fully Connected		| Outputs 120       							|
| RELU					|												|
| Drop out				| 50%											|
| Fully Connected		| Outputs 84       								|
| RELU					|												|
| Fully Connct'd Output	| Outputs 43									|
 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the ninth cell of the ipython notebook. 

To train the model, I used an AdamOptimizer along with a batch size of 128, a learning rate of 0.005, and 10 epochs.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eleventh cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.977
* validation set accuracy of 0.934 

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? A standard LeNet was chosen to start with
* What were some problems with the initial architecture? The 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The 5 sample images ca be seen in the 5(underscore)sample directory
After looking at the images, I'm having a difficult time understanding the predictions of the model.  In the first case, the 20kmph sign was identified as a "turn right ahead".  Frankly, even though the validation and test sets came in at 0.934 accuracy, I'd still like to try different architectures.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)	| Turn Right Ahead 								| 
| Children Crossing		| Turn Left Ahead 								|
| No Entry				| No Entry										|
| Turn Right Ahead		| Speed limit (30km/h)			 				|
| Stop					| Stop      									|


The model was able to correctly guess only 2 of the 5 traffic signs, which gives an accuracy of 40%. This does not compare favorably to the accuracy on the test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the twelth cell of the Ipython notebook.

For the first image, the model is sure that this is a "Turn Right Ahead" sign (probability of 0.95), but the image is a "Speed limit (20km/h)". The top five soft max probabilities are below and correct one was not in the top 5

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Turn Right Ahead								| 
| .04     				| Stop 											|
| <.01					| Keep Left										|
| <.01	      			| Turn Left Ahead				 				|
| <.01				    | No Vehicles      								|


For the second image, the model is sure that this is a "Turn Left Ahead" sign (probability of 0.91), but the image is a "Children Crossing". The top five soft max probabilities are below and correct one was #2, but a distant second

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .92         			| Turn Right Ahead								| 
| .03     				| Children Crossing								|
| .01					| End of All Speed and Passing Limits			|
| <.01	      			| Beware of ice/snow			 				|
| <.01				    | Keep Right      								|


For the third image, the model is sure that this is a "No Entry" sign (probability of >0.99) and that is correct. The top five soft max probabilities are below

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No Entry										| 
| <.01     				| Turn Left Ahead								|
| <.01					| Stop											|
| <.01	      			| Beware of ice/snow			 				|
| <.01				    | Ahead Only      								|


For the fourth image, the model is not confident that this is a "Speed limit (30km/h)" sign (probability of 0.45) and it should not be because the sign is actuall a "Turn Right Ahead". The top five soft max probabilities are below and the correct answer was third.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .45         			| Speed limit (30km/h)							| 
| .15     				| Speed limit (70km/h)							|
| .13					| Turn Right Ahead								|
| .09	      			| Speed limit (80km/h)			 				|
| .06				    | Roundabout Mandatory      					|


For the fifth image, the model is confident that this is a "Stop" sign (probability of 0.99) and it is a Stop sign. The top five soft max probabilities are below.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop											| 
| <.01     				| Turn Right Ahead								|
| <.01					| Speed limit (30km/h)							|
| <.01	      			| Wild Animals Crossing			 				|
| <.01				    | No Entry     									|





