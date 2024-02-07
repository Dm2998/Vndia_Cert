# Vndia_Cert
lab_7


Assessment¶
Congratulations on going through today's course! Hopefully, you've learned some valuable skills along the way and had fun doing it. Now it's time to put those skills to the test. In this assessment, you will train a new model that is able to recognize fresh and rotten fruit. You will need to get the model to a validation accuracy of 92% in order to pass the assessment, though we challenge you to do even better if you can. You will have the use the skills that you learned in the previous exercises. Specifically, we suggest using some combination of transfer learning, data augmentation, and fine tuning. Once you have trained the model to be at least 92% accurate on the validation dataset, save your model, and then assess its accuracy. Let's get started.

The Dataset
In this exercise, you will train a model to recognize fresh and rotten fruits. The dataset comes from Kaggle, a great place to go if you're interested in starting a project after this class. The dataset structure is in the data/fruits folder. There are 6 categories of fruits: fresh apples, fresh oranges, fresh bananas, rotten apples, rotten oranges, and rotten bananas. This will mean that your model will require an output layer of 6 neurons to do the categorization successfully. You'll also need to compile the model with categorical_crossentropy, as we have more than two categories.

Image
Load ImageNet Base Model
We encourage you to start with a model pretrained on ImageNet. Load the model with the correct weights, set an input shape, and choose to remove the last layers of the model. Remember that images have three dimensions: a height, and width, and a number of channels. Because these pictures are in color, there will be three channels for red, green, and blue. We've filled in the input shape for you. This cannot be changed or the assessment will fail. If you need a reference for setting up the pretrained model, please take a look at notebook 05b where we implemented transfer learning.

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
from tensorflow import keras
​
#base_model = keras.applications.VGG16(
 #   weights=FIXME,
  #  input_shape=(224, 224, 3),
   # include_top=FIXME)
​
base_model = keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)
Freeze Base Model
Next, we suggest freezing the base model, as done in notebook 05b. This is done so that all the learning from the ImageNet dataset does not get destroyed in the initial training.

# Freeze base model
# base_model.trainable = FIXME
base_model.trainable = False
Add Layers to Model
Now it's time to add layers to the pretrained model. Notebook 05b can be used as a guide. Pay close attention to the last dense layer and make sure it has the correct number of neurons to classify the different types of fruit.

# Create inputs with correct shape
# inputs = FIXME
​
# x = base_model(inputs, training=False)
​
# # Add pooling layer or flatten layer
# x = FIXME
​
# # Add final dense layer
# outputs = keras.layers.Dense(FIXME, activation = 'softmax')(x)
​
# # Combine inputs and outputs to create model
# model = keras.Model(FIXME)
​
inputs = keras.Input(shape=(224, 224, 3))
​
x = base_model(inputs, training=False)
​
# Add pooling layer or flatten layer
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
​
# Add final dense layer
outputs = keras.layers.Dense(1, activation = 'softmax')(x)
​
# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)
model.summary()
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
vgg16 (Model)                (None, 7, 7, 512)         14714688  
_________________________________________________________________
global_average_pooling2d (Gl (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 1)                 513       
=================================================================
Total params: 14,715,201
Trainable params: 513
Non-trainable params: 14,714,688
_________________________________________________________________
Compile Model
Now it's time to compile the model with loss and metrics options. Remember that we're training on a number of different categories, rather than a binary classification problem.

# model.compile(loss = FIXME , metrics = FIXME)
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=False), metrics=[keras.metrics.BinaryAccuracy()])
​
Augment the Data
If you'd like, try to augment the data to improve the dataset. Feel free to look at notebook 04a and notebook 05b for augmentation examples. There is also documentation for the Keras ImageDataGenerator class. This step is optional, but it may be helpful to get to 92% accuracy.

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
​
# datagen_train = ImageDataGenerator(FIXME)
# datagen_valid = ImageDataGenerator(FIXME)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
​
datagen_train = ImageDataGenerator( samplewise_center=True,  # set each sample mean to 0
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,)
datagen_valid = ImageDataGenerator(samplewise_center=True)
Load Dataset
Now it's time to load the train and validation datasets. Pick the right folders, as well as the right target_size of the images (it needs to match the height and width input of the model you've created). For a reference, check out notebook 05b.

# load and iterate training dataset
# train_it = datagen_train.flow_from_directory(
#     FIXME,
#     target_size=FIXME,
#     color_mode="rgb",
#     class_mode="categorical",
# )
# # load and iterate validation dataset
# valid_it = datagen_valid.flow_from_directory(
#     FIXME,
#     target_size=FIXME,
#     color_mode="rgb",
#     class_mode="categorical",
# )
​
from tensorflow.keras.preprocessing.image import ImageDataGenerator
​
​
# Load and iterate training dataset
train_it = datagen_train.flow_from_directory(
    "data/fruits/train/",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
)
​
# Load and iterate validation dataset
valid_it = datagen_valid.flow_from_directory(
    "data/fruits/valid/",  # Corrected path to the validation dataset
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
)
Found 1182 images belonging to 6 classes.
Found 329 images belonging to 6 classes.
Train the Model
Time to train the model! Pass the train and valid iterators into the fit function, as well as setting the desired number of epochs.

# model.fit(FIXME,
#           validation_data=FIXME,
#           steps_per_epoch=train_it.samples/train_it.batch_size,
#           validation_steps=valid_it.samples/valid_it.batch_size,
#           epochs=FIXME)
​
# model.fit(train_it,
#           validation_data=valid_it,
#           steps_per_epoch=train_it.samples // train_it.batch_size,
#           validation_steps=valid_it.samples // valid_it.batch_size,
#           epochs=10)
​
​
model.fit(train_it,
          # validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=20)
Epoch 1/20
37/36 [==============================] - 14s 382ms/step - loss: 1.6159 - binary_accuracy: 0.8153
Epoch 2/20
37/36 [==============================] - 6s 171ms/step - loss: 1.3882 - binary_accuracy: 0.8345
Epoch 3/20
37/36 [==============================] - 6s 152ms/step - loss: 1.2175 - binary_accuracy: 0.8440
Epoch 4/20
37/36 [==============================] - 6s 152ms/step - loss: 1.0791 - binary_accuracy: 0.8579
Epoch 5/20
37/36 [==============================] - 6s 151ms/step - loss: 0.9710 - binary_accuracy: 0.8737
Epoch 6/20
37/36 [==============================] - 6s 151ms/step - loss: 0.8805 - binary_accuracy: 0.8866
Epoch 7/20
37/36 [==============================] - 6s 153ms/step - loss: 0.8063 - binary_accuracy: 0.9000
Epoch 8/20
37/36 [==============================] - 6s 152ms/step - loss: 0.7456 - binary_accuracy: 0.9131
Epoch 9/20
37/36 [==============================] - 6s 149ms/step - loss: 0.6952 - binary_accuracy: 0.9215
Epoch 10/20
37/36 [==============================] - 6s 149ms/step - loss: 0.6501 - binary_accuracy: 0.9310
Epoch 11/20
37/36 [==============================] - 6s 150ms/step - loss: 0.6153 - binary_accuracy: 0.9373
Epoch 12/20
37/36 [==============================] - 5s 148ms/step - loss: 0.5833 - binary_accuracy: 0.9418
Epoch 13/20
37/36 [==============================] - 5s 145ms/step - loss: 0.5519 - binary_accuracy: 0.9461
Epoch 14/20
37/36 [==============================] - 5s 145ms/step - loss: 0.5293 - binary_accuracy: 0.9491
Epoch 15/20
37/36 [==============================] - 5s 148ms/step - loss: 0.5043 - binary_accuracy: 0.9521
Epoch 16/20
37/36 [==============================] - 5s 144ms/step - loss: 0.4836 - binary_accuracy: 0.9554
Epoch 17/20
37/36 [==============================] - 5s 147ms/step - loss: 0.4663 - binary_accuracy: 0.9567
Epoch 18/20
37/36 [==============================] - 5s 146ms/step - loss: 0.4494 - binary_accuracy: 0.9571
Epoch 19/20
37/36 [==============================] - 5s 147ms/step - loss: 0.4324 - binary_accuracy: 0.9595
Epoch 20/20
37/36 [==============================] - 5s 147ms/step - loss: 0.4193 - binary_accuracy: 0.9607
<tensorflow.python.keras.callbacks.History at 0x7fe770284ba8>
Unfreeze Model for Fine Tuning
If you have reached 92% validation accuracy already, this next step is optional. If not, we suggest fine tuning the model with a very low learning rate.

# Unfreeze the base model
base_model.trainable = FIXME
​
# Compile the model with a low learning rate
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = FIXME),
              loss = FIXME , metrics = FIXME)
model.fit(FIXME,
          validation_data=FIXME,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=FIXME)
Evaluate the Model
Hopefully, you now have a model that has a validation accuracy of 92% or higher. If not, you may want to go back and either run more epochs of training, or adjust your data augmentation.

Once you are satisfied with the validation accuracy, evaluate the model by executing the following cell. The evaluate function will return a tuple, where the first value is your loss, and the second value is your accuracy. To pass, the model will need have an accuracy value of 92% or higher.

model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)
11/10 [================================] - 2s 223ms/step - loss: 0.4362 - binary_accuracy: 0.9559
[0.4362468719482422, 0.9559269547462463]
Run the Assessment
To assess your model run the following two cells.

NOTE: run_assessment assumes your model is named model and your validation data iterator is called valid_it. If for any reason you have modified these variable names, please update the names of the arguments passed to run_assessment.

from run_assessment import run_assessment
run_assessment(model, valid_it)
Evaluating model 5 times to obtain average accuracy...

11/10 [================================] - 1s 134ms/step - loss: 0.4325 - binary_accuracy: 0.9554
11/10 [================================] - 2s 137ms/step - loss: 0.4295 - binary_accuracy: 0.9574
11/10 [================================] - 2s 141ms/step - loss: 0.4189 - binary_accuracy: 0.9600
11/10 [================================] - 2s 138ms/step - loss: 0.4465 - binary_accuracy: 0.9554
11/10 [================================] - 1s 131ms/step - loss: 0.4115 - binary_accuracy: 0.9620

Accuracy required to pass the assessment is 0.92 or greater.
Your average accuracy is 0.9581.

Congratulations! You passed the assessment!
See instructions below to generate a certificate.
