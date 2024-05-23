import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
import argparse


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Read argument options
parser = argparse.ArgumentParser(prog='PTC', description='papillary thyroid carcinoma')
parser.add_argument('-m', type=str, help='[mobile, resnet, efficient]')
parser.add_argument('-n', type=int, help='number of classes [2, 3]')
# parser.add_argument('-k', type=int, help='fine-tune from this layer onwards')
args = parser.parse_args()

# 0. Load and split data
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

def get_directory_data(n):
    if n == 2:
        return "dat/2/"
    else:
        return "dat/3/"

directory = get_directory_data(args.n)
train_dataset = image_dataset_from_directory(directory, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, validation_split=0.2, subset='training', seed=42)
validation_dataset = image_dataset_from_directory(directory, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, validation_split=0.2, subset='validation', seed=42)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)


# 1. Load preprocess function
def preprocess(model_type):
    match model_type:
        case "mobile":
            return tf.keras.applications.mobilenet_v2.preprocess_input
        case "resnet":
            return tf.keras.applications.resnet50.preprocess_input
        case "efficient":
            return tf.keras.applications.efficientnet.preprocess_input
        case _: 
            return tf.keras.applications.mobilenet_v2.preprocess_input

preprocess_input = preprocess(args.m)

# 2. Load a base model
def get_base_model(model_type):
    match model_type:
        case "mobile":
            return tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        case "resnet":
            return tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        case "efficient":
            return tf.keras.applications.EfficientNetB7(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        case _:
            return tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

IMG_SHAPE = IMG_SIZE + (3,)
base_model = get_base_model(args.m)

base_model.summary()

# 3. Create a model from a base model in which all layers are freezed

def thyroid_model(num_classes=2):
    ''' Define a tf.keras model for binary classification out of a base model
    Arguments:
        image_shape -- image width and height
        base_model -- the base model
        num_classes -- number of Bx class
    Returns:
        tf.keras.model
    '''
    
    # freeze the base model by making it non trainable
    base_model.trainable = False

    # create the input layer (same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    
    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(inputs) 
    
    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False)
    
    # use global avg pooling to summarize the info in each channel
    x = tf.keras.layers.GlobalAveragePooling2D()(x) 
    # include dropout with probability of 0.2 to avoid overfitting
    x = tf.keras.layers.Dropout(0.2)(x)
        
    # use a prediction layer
    outputs = tf.keras.layers.Dense(num_classes)(x)
        
    model = tf.keras.Model(inputs, outputs)
    
    return model

ptc = thyroid_model(2)
ptc.summary()

# 4. Compile and train the freeezed model for 20 epochs:

base_learning_rate = 0.0001
loss_function=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
ptc.compile(optimizer=optimizer, loss=loss_function,  metrics=['accuracy'])

initial_epochs = 20
history = ptc.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

# 5. Plot the training and validation accuracy

acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.yticks(np.arange(0, 1, step=0.1))
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.5])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('freezed-%s.png' % args.m)

# 6. Fine-tune the model
base_model = ptc.layers[3] # inputs -> preprocess -> base_model -> GlobalAveragePooling2D -> Dropout -> Dense
base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = int(0.8 * len(base_model.layers))
print("Fine-tune from layer: ", fine_tune_at)

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    
# Define an Adam optimizer with a learning rate of 0.1 * base_learning_rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1*base_learning_rate)
ptc.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

fine_tune_epochs = 15
total_epochs =  initial_epochs + fine_tune_epochs

history_ext = ptc.fit(train_dataset, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=validation_dataset)

acc += history_ext.history['accuracy']
val_acc += history_ext.history['val_accuracy']

loss += history_ext.history['loss']
val_loss += history_ext.history['val_loss']

# 7. Plot the fine-tuned summary

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.yticks(np.arange(0, 1, step=0.1))
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 2.0])
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine-Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig("tuned-%s.png' % args.m")

# 8. Performance on the validation set

z_test = ptc.predict(validation_dataset, batch_size=32)
prediction = tf.math.argmax(z_test, axis=1)
y_test = tf.concat([y for (_, y) in validation_dataset], 0)
R = tf.math.confusion_matrix(y_test, prediction)
print(R)

