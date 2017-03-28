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


## To make it easier to determine what has changed from my first submission to this second one I have summarized the updates here

1. A typo was corrected in the normalization.  The min should have been .1 and it was .2.  This was corrected.
2. L2 regularization was added in the loss(underscore)operation with a beta of .01
3. Perturbing the images in rotation was reduced from up to 15 degrees (CW or CCW) to up to 5 degrees
4. Validation and test set accuracy was improved to 0.935 from 0.934
5. Training set accuracy decreased from 0.977 to 0.949
5. BIG change in the 5 sample images from the web.  Accuracy here jumped from 40% to 80% and is now in line with what might be expected with a small sample set.
6. Tables detailing results for the 5 samples has been updated in the main body of this document

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

To train the model, I used an AdamOptimizer along with a batch size of 128, a learning rate of 0.005, and 10 epochs along with L2 regularization using a beta of 0.01

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eleventh cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.949
* validation set accuracy of 0.935 

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? A standard LeNet was chosen to start with
* What were some problems with the initial architecture? Not much, some minor changes to the final outputs matrices was all that was required.
* How was the architecture adjusted and why was it adjusted? The model was adjusted to include regularization and a drop out of 50% just before the 2nd fully connected layer.
* Which parameters were tuned? How were they adjusted and why? The batch size and learning rate were changed MANY times.  After testing with several learning rates (.1, .05, .01, .005, .001) and batch sizes (64, 128 256), .005 and 128 yielded the best results.  The number of epochs was also varied, starting with 1, then 3, 10, and 15.  Ten was eventually settled on.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application? If was set up for it already (32x32x1) and by teweaking a few parameters I felt I could get it to work well
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Model has accuracies of .949/.935/.935 for training/validation/testing sets and scored 80% on the small 5 sample images from the web (and the 1 it got wrong was its second choice).
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The 5 sample images ca be seen in the 5(underscore)sample directory
After looking at the images, I'm having a difficult time understanding the predictions of the model.  In the first case, the 20kmph sign was identified as a "turn right ahead".  Frankly, even though the validation and test sets came in at 0.934 accuracy, I'd still like to try different architectures.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)	| Speed limit (20km/h)							| 
| Children Crossing		| Beware of ice/snow							|
| No Entry				| No Entry										|
| Turn Right Ahead		| Turn Right Ahead				 				|
| Stop					| Stop      									|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compare sfavorably to the accuracy on the validation and test sets given the small sample size.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the twelth cell of the Ipython notebook.

For the first image, the model is confident that this is a "TSpeed limit (20km/h)" sign (probability of 0.76), and that is correct. The top five soft max probabilities are below.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .76         			| Speed limit (20km/h)							| 
| .15     				| Speed limit (60km/h)							|
| .03					| Keep Left										|
| .03	      			| Speed limit (70km/h)			 				|
| <.01				    | Turn Right Ahead 								|


For the second image, the model is pretty sure that this is a "Beware of ice/snow" sign (probability of 0.64), but the image is a "Children Crossing" which came in second at 0.16. After visually examing 10 samples of the training set for "Beware of ice/snow" I can definitely understand the difficulty the model might have with this sign as all 10 were essentially blurry black blobs in a red triangle which is pretty close to what a "Children Crossing" sign looks like. The top five soft max probabilities are below. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .64         			| Beware of ice/snow							| 
| .16     				| Children Crossing								|
| .13					| Right-of-way at the next intersection			|
| .02	      			| End of no passing			 					|
| .02				    | Bicycles crossing      						|


For the third image, the model is sure that this is a "No Entry" sign (probability of >0.99) and that is correct. The top five soft max probabilities are below.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| >.99         			| No Entry										| 
| <.01     				| Stop											|
| <.01					| No passing									|
| <.01	      			| Yield			 								|
| <.01				    | End of no passing								|


For the fourth image, the model is very confident that this is a "Turn Right Ahead" sign (probability of >0.99) and it should not be because the sign is actually a "Turn Right Ahead". The top five soft max probabilities are below.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| >.99        			| Turn right ahead								| 
| <.01     				| Stop											|
| <.01					| Go straight or right							|
| <.01	      			| Ahead only	 								|
| <.01				    | Right-of-way at the next intersection			|


For the fifth image, the model is unsure that this is a "Stop" sign (probability of 0.99), but it is the model's best guess and it is correct. The top five soft max probabilities are below.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .43         			| Stop											| 
| .10     				| Speed limit (30km/h)							|
| .06					| Turn right ahead								|
| .06	      			| Speed limit (50km/h)			 				|
| .04				    | Priority road									|





