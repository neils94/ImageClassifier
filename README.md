# ImageClassifier
- To summarize: this Image Classifier uses a pytorch model (vgg16) to classify over 100 different classes of flowers.

# Methodology
- Preprocessing: First transformations are applied to normalize the images in a way that torch models can recognize them, as well as center crop and resize the images in order to extract the main features.  Next, the image data is split into 3 separate folders; one for training, one for testing and one for validation. After this, a json package is used to decode a file that takes the real flower labels matching to their real class numbers in order to use the file later to measure the predicted vs actual values for python to interpret it.
 
- Model selection: For this type of an image classification problem either one of resnet, vgg, alexnet models would have worked equally as well but I decided to work with vgg16. 
 
- Transfer learning: In order to successfully have a model that can train on the exact dataset, the classifier part of the VGG model has to be replaced to match the output space of the dataset. (I also added some dropout layers to lessen over/underfitting.) 
 
- Model Training: Now that the model is ready to be trained; the loss function that is used is Negative Log Likelihood loss (suitable for image classification) and Adam is used to optimize only the classifiers parameters. This is important to not optimize the entire models parameters to avoid memory overload in training. The validation set is also used in the training loop with all gradients turned off to measure if the model is correctly training without overfitting. This way we can test our model per loop to see if we see imrpovements on new images that the model isn't training on. 

- Model testing: Similar to the concept of validation the model will now try to generalize to more images with all gradients turned off. If the model is trained well, using this new dataset the model should have an accuracy around 70% on new images.
 
- Saving Model: The model is now trained and tested multiple new imagesets, after this point the model in its current state (model parameters, optimizer, loss function) is saved as a checkpoint. The checkpoint allows us to use the fully trained model instead of having to re-train it for every use. 

- Further inference: Now comes the ultimate test, the model has been trained and tested on multiple imagesets. If our model is successful it should be able to take any single image and given a log likelihood provide us with the most likely flower name to that image. This is tested by creating a function to normalize a single image and putting that single image through the model in order to graph the log likelihood of the names (labels) chosen by the model to try to match the input.


# Results
- At the bottom of the .ipynb file you will be able to see a single image processed and displayed. The graph displays the correct flower (blanket flower) had the highest probability of being chosen. The remaining flowers each had <10% probability of being chosen by the model.

- ![ICResults](https://user-images.githubusercontent.com/63812857/132139948-a13ab1a3-f73f-4521-a2ea-d093b8ec75ea.png)


