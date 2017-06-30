# Self-Driving-Car-Simulator-Code
This Project includes Udacity simulator and code for self drive.

## Overview
In the Project of Udacity Self-Driving car nanodegree (www.udacity.com), we’re invited to design a system that drives a car autonomously in a simulated environment.The objective of this project is to clone human driving behavior using a Deep Neural Network. In order to achieve this,I am going to use a simple Car Simulator. During the training phase, I will navigate car inside the simulator using the keyboard.The Image is attached below of the training track


### Dataset Gathering
While the car is being driven it records all the data of the training phase into a file called "Training.csv".
This file stores 3 images:
CENTER CAMERA IMAGE,
LEFT CAMERA IMAGE,
RIGHT CAMERA IMAGE,
with 4 other varibales :
STEERING ANGLE,
THROTTLE,
SPEED,
REVERSE.So now Inputs: Recorded images and steering angles, Outputs: Predicted steering angles.
### UDACITY STIMULATOR
The data to be gathered is collected through beta stimulator provided by Udacity.You have to explicitly make a folder named
"DATA" and leave rest to the stimulator.The stimulator will itslef create a "driving_log.csv" file with images.
#### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Keras=2.0.3](https://keras.io/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [TensorFlow](http://tensorflow.org)
- [Pandas](http://pandas.pydata.org/)
- [OpenCV](http://opencv.org/)
- [Matplotlib](http://matplotlib.org/) 
- [Jupyter](http://jupyter.org/) 
- [numpy]
- [wheel]
- [utlis]
##### How to Run the Model

This repository comes with trained model which you can directly test using the following command.

- "python drive.py model.h5"(For this include a data folder containing the driving_log.csv file,See the section UDACITY STIMULATOR above)


- "python drive.py model_x.h5" (This includes the already trained model on my attempt for better results)


- "Python model.h5"            (To train a model of your own)


###### Architecture
Image recognition and decision of steering angle

For us, it is almost impossible not to see an image and recognize what is inside it. But, for a computer, a image is just a bunch of numbers. More specifically, it is an array of numbers for 0 to 255, in the three channels (Red, Green, Blue), for each pixel of the image.

![](https://cdn-images-1.medium.com/max/800/1*WU3SG3ZeZxGb8vnAJLlqQw.png)



What are the relevant features of a image to decide the steering angle?
Are the clouds in the sky relevant? Is the river important? Are the lanes important? Or the colors of the road? What happens in the part of the circuit where there are not lines (like the dirty exit below?). These are trivial questions for humans, but a machine must be trained to consider the right features.

![](https://cdn-images-1.medium.com/max/800/1*VLuZ-pHIoYy-kY7cQfpxhQ.png)


The unpaved road to the right  cause me a lot of trouble…


###### The approach and tools

There are several possible methods to try to solve the problems above. The approach suggested by Udacity was to use image processing and deep neural networks.


The tools used were Keras (a wrapper running over Tensor Flow), OpenCV (to do some image transformations), numpy (matrix operations) and python (to put all these things together).


###### Analysis of the images

The first input are images of a camera inside the car. There are three cameras: center, left and right.

![](https://cdn-images-1.medium.com/max/600/1*pyYePAXeavIA1ZNfpVQanQ.jpeg)
![](https://cdn-images-1.medium.com/max/600/1*e1xgFWD1n-EmzJpBWfJCaw.jpeg)

Example of center and left cameras 
 The second input is a csv file, each line containing the name of the correspondent image, steering angle, throttle, brake.
 
 The steering angle is normalized from -1 to 1, corresponding in the car to -25 to 25 degrees.
 
The data provided by Udacity has 8036 images from each camera, resulting in 24108 images. In addition to it, we can generate our own data.


The bad news. Just input raw images to the network will not work.


Most of the images has the equivalent of zero steering angle. The curves are misrepresented, because most of time the car goes straight.


The network will not converge to a good solution. We have to preprocess the data: to do what is called “image augmentation”.


###### Center and lateral images


So, we have data from lateral cameras. But what to do with it?


Following a suggestion from the great carND forum (by the way, almost all of these tips are from there), I added a correction angle of 0.10 to the left image, and -0.10 to the right one. The idea is to center the car, avoid the borders.
 
























###### Resize


Because of computational limits, it is a good thing to resize the images, after cropping it. Using the size of NVIDIA paper (66 x 200 pixels) my laptop went out of memory. I tried (64 x 64), (64 x 96), (64 x 128)…


I also reduced the size of the stride of the convolutional layers. Since the image is smaller, the stride can also be smaller.

![](https://cdn-images-1.medium.com/max/800/1*0Z3wZZ0SprS61V0RMrhYeg.png)






X_in[i,:,:,0] = cv2.resize(imgScreenshots[i].squeeze(), (size2,size1))
Some effects of resizeAll of these sizes work. An curious effect. When the image is smaller, the zig zag of the car is greater. Surely because there are fewer details in the image.
 
######  Crop 






The image was cropped to remove irrelevant portions of the image, like the sky, or the trees. Doing this, we’re assuming the camera is fixed in a stable position.

![](https://cdn-images-1.medium.com/max/800/1*u-l8ti07RJNfT6s9yc86NQ.png)







Effect of crop

Because inside the model a image is just a matrix, a numpy command easily do this operation.

crop_img = imgRGB[40:160,10:310,:] 



Throwing away to superior portion of the image and 10 pixels from each side of the original image = (160, 320)


It makes sense to do the transformations (like the lateral shift) before the cropping, since we’re losing information. Doing the opposite, we will feed the model with an image with a lateral black bar.


 
I used the conversion command of opencv to transform the image in RGB.


 imgOut = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

![](https://cdn-images-1.medium.com/max/800/1*bwzxzc4I0Ue4p6LTjhkJ4Q.png)





The model used full color RGB. In addition to it, I used a first layer of the NN that just transform the 3 color channels in 1 channel, with the ponderation Keras chooses, instead of me choosing the right channel.
 



 model.add(Convolution2D(1,1,1, border_mode = ‘same’,init =’glorot_uniform’,input_shape=(size1,size2,3)))


Normalization is made because the neural network usually works with small numbers: sigmoid activation has range (0, 1), tanh has range (-1,1).


Images have 3 channels (Red, Green, Blue), each channel has value 0 to 255. The normalized array has range from -1 to 1.
 


 X_norm = X_in/127.5–1


######  Modeling
 
 
 
Once the image processing and augmentation is done, it is time to design the Neural Network (NN) . But there are an infinite number of architectures of neural networks.

As a starting point, I used the NVIDIA End-to-End model (https://arxiv.org/abs/1604.07316), that has the configuration described below.






![](https://cdn-images-1.medium.com/max/800/1*YY8LNITxGOo37NQD05mJqA.png)










Given the features, how to decide about the steering angle?

It is done by fully connected layers after the convolutional ones.
The matrix is flattened, because the spatial information of rows and columns doesn’t matter any longer. The NN is like a function that is feed with images, and for each image, give the steering angle as steering angle, which is compared to the angle of the human driver.





###### Loss error function

How to compare the predicted and actual angles?

model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=’mse’)






 
 
###### Dropout
A dropout layer, with probability of 50%, was used, after the first full connected layer.
The idea of the dropout layer is very interesting. The network must be robust enough to do the forecast (or to clone the behavior) even without some of this connections. It is like trying to recognize an image just looking at parts of it. The 50% says only random half of the neurons in that layer will be kept in a fitting step. If it were 20%, it will keep 80% of the neurons in Keras. In TensorFlow, it is the opposite, 20% will be kept.


It is good to avoid overfitting. The network can be memorizing wrong features of the image. Memorize the trees of the road is a bad generalization. Dropout helps to avoid it.


Why dropout just the first full conected layer? And why 50%? Actually, there are several other possible solutions. There is no right single answer, a lot of this is empirical.


###### Learning rate


A learning rate of 0.001 was used, with Adam optimizer. Adam optimizer is a good choice in general, because it fine tunes the learning rate automatically.


ELU activation function proved superior. This paper discusses that ELU is better than RELU, a function that avoids vanishing gradient but keeps the non-linearity of the network. (http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/).





###### RESULTS AND CONCLUSIONS
In the initial stage of the project, I used a dataset generated by myself. That dataset was small and recorded while navigating the car using the laptop keyboard.
hen it comes to extensions and future directions, I would like to highlight followings.
* Train a model in real road conditions. For this, we might need to find a new simulator.
* Experiment with other possible data augmentation techniques.
* When we are driving a car, our actions such as changing steering angles and applying brakes are not just based on instantaneous driving decisions. In fact, curent driving decision is based on what was traffic/road condition in fast few seconds. Hence, it would be really interesting to seee how Recurrent Neural Network (**RNN**) model such as **LSTM** and **GRU** perform this problem.
* Finally, training a (deep) reinforcement agent would also be an interesting additional project.
###### LINKS
Udacity: https://www.udacity.com/


Tensor Flow: https://www.tensorflow.org/


Keras: http://keras.io/


Numpy, Scikit Learn: http://scikit-learn.github.io/stable


NVIDIA End-to-End: https://arxiv.org/abs/1604.07316


Le Cun paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf


ELU: http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/


Dropout: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
###### Credits
https://github.com/upul/Behavioral-Cloning
https://github.com/asgunzi/CarND-Simulator
