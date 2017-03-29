# Traffic Sign Recognition

## Files Submitted


[//]: # (Image References)

[image1]: doc_imgs/hist.png "Class Distribution"
[image2]: doc_imgs/gray.png "LeNet Preprocessing"
[image3]: doc_imgs/yuv.png "NVIDIANet Preprocessing"
[image4]: doc_imgs/errors.png "Incorrect Guesses"
[conv1]: doc_imgs/conv1.png "Conv. 1"
[conv2]: doc_imgs/conv2.png "Conv. 2"
[conv3]: doc_imgs/conv3.png "Conv. 3"
[conv4]: doc_imgs/conv4.png "Conv. 4"
[conv5]: doc_imgs/conv5.png "Conv. 5"
[image5]: doc_imgs/hist1.png "Class Distribution (Validation data)"
[image6]: doc_imgs/hist2.png "Class Distribution (Test data)"
[5img1]: doc_imgs/5img1.png "New Images"
[5img]: doc_imgs/5img.png "New Images Predictions (LeNet)"
[5img2]: doc_imgs/5img2.png "New Images Predictions (NVIDIANet)"


The submission contains the following files:
1. `Traffic_Sign_Classifier.ipynb` - Jupyter Notebook document containing the code.
2. `Traffic_Sign_Classifier.html` - HTML version of the notebook file.
3. Model-related files for LeNet and NVIDIANet models. They contain the weights of these
   models and can be loaded directly with `tf.Saver().restore()` command.

## Dataset Summary and Exploration

In this project, I create a deep learning classifier capable to label Germany street signs. The dataset was downloaded from [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and it contains the following characteristics (the code for this is available in the fourth code block of the Jupyter script):

- Number of training examples: 34799
- Number of validation examples: 4410
- Number of testing examples: 12630
- Shape of a traffic sign image is 32x32px
- Number of unique classes in the dataset: 43 (See Apendix 1 of the code for more details)

Class distribution of the entire dataset:

![hist][image1]

Looks like several classes occur more often than others. Here is a list of sign names
with 10 most occurences:


|    | ID | Sign Name                                    | Count |
|:--:|:--:|:--------------------------------------------:|:-----:|
| 1  | 2  | Speed limit (50km/h)                         | 3000  |
| 2  | 1  | Speed limit (30km/h)                         | 2940  |
| 3  | 13 | Yield                                        | 2880  |
| 4  | 12 | Priority road                                | 2790  |
| 5  | 38 | Keep right                                   | 2760  |
| 6  | 10 | No passing for vehicles over 3.5 metric tons | 2670  |
| 7  | 4  | Speed limit (70km/h)                         | 2640  |
| 8  | 5  | Speed limit (80km/h)                         | 2490  |
| 9  | 25 | Road work                                    | 1980  |
| 10 | 9  | No passing                                   | 1950  |

## Design and Test a Model Architecture

I initially used a small 5-layer Convolutional Network ([LeNet-5 from this classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81)), but the validation accuracy did not manage to get to the required 0.93, so 
I switched to using the 7-layer architecture from [NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and it achieved the required 0.93 accuracy.

### Preprocessing

#### LeNet Preprocessing

The preprocessing steps done was grayscaling followed by normalization. Grayscaling was done to shrink the input space to help the model learn with relatively low amount of data.

Normalization was done so the gradient descend may converge faster and in a more stable manner.

Here is an example of an image before and after this preprocessing step. Notice how the grayscaled version has more defined characteristics (e.g. the white sections are more white):

![LeNet Preprocessing][image2]

See the code blocks under the section **Pre-process the Data Set (normalization, grayscale, etc.)** for the script used for this section.

#### NVIDIANet Preprocessing

Prior to using NVIDIANet, there are three preprocessing steps done to the images:

1. Set color space to YUV. The paper/article from NVIDIA mentioned about using [YUV color space](https://en.wikipedia.org/wiki/YUV). Here is an interesting excerpt from its description:

   > It encodes a color image or video taking human perception into account, allowing reduced bandwidth for chrominance components, thereby typically enabling transmission errors or compression artifacts to be more efficiently masked by the human perception than using a "direct" RGB-representation.

2. Resize to 200x200 pixels. This is to ensure the input space is supported by the architecture.
3. Normalize.

Here is how the images look like when they are preprocessed. The first three sets (9 images on the left section) show Y, U, and V components respectively. The middle three images show the image converted back into RGB, and the last three images on the right section show completely processed images (notice the images were resized there, and they are placed diagonally to better present the x and y-axes tick marks.


![NVIDIA Preprocessing][image3]

On a hindsight, it looks like only the Y layer of YUV color space is needed. For this project I used the entire three layers, but this might be an area of further research.

The code for this section is available under the section "Update the convolutional network" in the notebook document.

### Model Architecture

I followed the architecture in the [article/paper](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) exactly as it is, but decided not to augment the dataset since the final algorithm's performance is good enough to pass the specification required in this project. I intend to improve on this in a future project.

The code for the LeNet model is located in the `LeNet` function, and the final model in the `NVIDIANet` function.

My final model consisted of the following layers:

| Layer                     |     Description                               | 
|:-------------------------:|:---------------------------------------------:| 
| Input                     | 200x200x3 YUV image | 
| Convolution 5x5           | 2x2 stride, valid padding, outputs 98x98x24 |
| RELU                      |                       |
| Convolution 5x5           | 2x2 stride, valid padding, outputs 47x47x36 |
| RELU                      |                       |
| Convolution 5x5           | 2x2 stride, valid padding, outputs 22x22x48 |
| RELU                      |                       |
| Convolution 3x3           | 1x1 stride, valid padding, outputs 20x20x64 |
| RELU                      |                       |
| Convolution 3x3           | 1x1 stride, valid padding, outputs 18x18x64 |
| RELU                      |                       |
| Fully connected (flatten) | outputs 20736 |
| Fully connected           | outputs 100 |
| Fully connected           | outputs 43 |

The paper uses Fully connected 1164 -> 100 -> 50 -> 10 neurons, but I used 20736 -> 100 -> 43 since we have more output logits. Some more research can be done to find the optimal arrangement.

### Model Training

The code for training the model is located under the section **Train, Validate and Test the Model** for each LeNet and NVIDIANet sections.

I used 10 epochs, 128 batch size, and a learning rate of 0.001 in both architectures. Softmax Cross Entropy (`tf.nn.softmax_cross_entropy_with_logits`) was used as the loss score to reduce in each epoch.

### Solution Approach

The final Model uses the NVIDIANet architecture with the following settings:

- Epoch: 10
- Batch Size: 128
- Learning Rate: 0.001

Just to be entirely sure with the result, we have also tested the model on a set of images it has never seen previously i.e. the test set. The following are the result of using the final model to predict both Validation and Test sets:
- **Training Accuracy: 0.998** (Code is available under **Investigating incorrectly predicted training images** section)
- **Validation Accuracy: 0.941** (Code is available under **Train, Validate and Test the Model** section)
- **Testing Accuracy: 0.932** (Code is available under **Performance of the final model on the test set** section)

#### What was the first architecture that was tried and why was it chosen?

LeNet was the first architecture tried because it was simple and readily available from the classroom.

#### What were some problems with the initial architecture?

Validation score seems to revolve around 0.85 to 0.92 which was below the requirement to pass this project (0.93 validation score)

Below are images the initial model failed to predict and their respective correct answers:

![errors][image4]

Some errors happened to predictions that were supposed to be "obvious" like "Speed limit (30km/h)".

#### How was the architecture adjusted and why was it adjusted?

The architecture was adjusted by simply increasing the number of neurons in each layer and adding more layer. Another interesting adjustment that was made was in removing the Pooling layers. It is well known for a deep learning network to work better by deepening the architecture, so that was what I did. To get a sense of guideline I intended to start with NVIDIANet and improve it further, but turned out the validation accuracy was good enough as it was.

#### Which parameters were tuned? How were they adjusted and why?

Experiments were done only on LeNet architecture since it took about 30 seconds compared to 9-10 minutes with NVIDIANet architecture. There were a couple of interesting pattern I have observed here:

- Fewer epochs led to much lower accuracies but it does not get better after 10 epochs.
- Larger batch sizes led to lower accuracies but faster training. On earlier epochs the accuracies could get as low as 0.1 to 0.4 when using a batch size of 512 compared to 0.6 with a batch size of 128.
- Learning rate of 0.001 was perfect at least for this dataset.

Here are the results of some experiments using LeNet architecture:

| Epoch | Batch Size | Learning Rate | Validation Accuracy | Training Time |
|:-----:|:----------:|:-------------:|:-------------------:|:-------------:|
| 10 | 128 | 0.001 | 0.919 | 20.1s |
| 10 | 512 | 0.001 | 0.853 | 15.6s |
| 12 | 128 | 0.001 | 0.886 | 23.2s |
| 10 | 128 | 0.003 | 0.916 | 19.8s |
| 10 | 128 | 0.0005 | 0.886 | 20 .2s |

#### What are some of the important design choices and why were they chosen?

The most important design choice was obviously the change to a deeper neural network for the reason explained above (deeper = better). Convolutional layers add more features by including more and more subtle characteristics as they go deeper. As an example, here are the outputs of convolutional layers in the NVIDIANet model:

##### Convolution Layer 1

![conv1][conv1]

##### Convolution Layer 2

![conv2][conv2]

##### Convolution Layer 3

![conv3][conv3]

##### Convolution Layer 4

![conv4][conv4]

##### Convolution Layer 5

![conv5][conv5]

Notice how features are more variative and "blurry" as the layer gets deeper.

#### What architecture was chosen?

The architecture explained in [this NVIDIA article](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and the research paper linked to it.

#### Why did you believe it would be relevant to the traffic sign application?

Because the structure looks similar to the already working LeNet architecture.

#### How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

We can definitely believe the result from predicting the test set more than validation test, since it was done on the data the final model has never seen before. However, as shown in the class distribution above, here is the histogram again:

![hist][image1]

The classes are not equally distributed. It looks like the same trend is happening on both Validation and Test datasets:

![hist][image5]
![hist][image6]


Due to this imbalance amount of data, there are two things that I would improve:
1. Use precision and recall instead of accuracy score. F1 is a good candidate since it is a harmonic mean between precision and recall.
2. Use a stratified shuffle split cross validation technique during the training-validation step.
3. Augmenting the dataset by duplicating less popular images and injecting noise into them.

Again, I shy away from doing these improvements for this particular project since the result is good enough to pass the specification, but this is something I would like to revisit in the future after completing this Self-Driving program.

## Test a Model on New Images

### Acquiring New Images

These are the five additional images I fed into the model 

![5img][5img1]

The first image might be difficult to classify due to the lighting of the entire image. A similar problem exists on the last image, but this time I tried to use a picture where the contrast is higher in the background, to see if that may confuses the model.

The second image is of medium difficulty. The contrast is not really big, but we can see the characteristics clearly.

The third and fourth images are the easiest, but they have some noises in the background. I'd like to see whether the model was able to differentiate background image from the sign.

The code block that outputs these images is located under section "Load and Output the Images".

### Performance on New Images

The model was able to predict all five images correctly. Interestingly, although both LeNet and NVIDIANet made 100% correct predictions, their softmax probabilities tell some nuances within these predictions.

### Model Certainty - Softmax Probabilities

Following are each model's softmax probabilities (i.e. how sure was the model that each of these predictions was correct)

#### LeNet Softmax Probabilities
![5img][5img]

The code is located under section "Output Top 5 Softmax Probabilities For Each Image Found on the Web".

#### NVIDIANet Softmax Probabilities
![5img][5img2]

The code is located under section "Softmax Probabilities for 5 Images from the Web".

Notice how LeNet had a tiny doubt on two of the hardest images above (99.97% and 99.91% confidence respectively), while the stronger model was totally confident in all of its predictions.