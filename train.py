import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam


def model_vgg16(img_size,num_classes,epoch):
    model = Sequential()
    model.add(VGG16(input_shape=(img_size,img_size,3), input_tensor=None,include_top=False,weights='imagenet'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    opt = Adam(lr=0.001, decay=0.001 / epoch)
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["acc"])
    return model


def model_lenet(img_size):
    model=Sequential()
    model.add(Conv2D(filters=6,kernel_size=5,input_shape=(img_size,img_size,1),padding='same',activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=16,kernel_size=5,activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(120,activation='relu'))
    model.add(Dense(84,activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model


batch_size = 8
img_size = 224
num_classes = 4
epoch = 50


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)


x_train = train_datagen.flow_from_directory(
        r'/home/pavithran/StonePaperScissor/dataset_split/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical')

x_test = test_datagen.flow_from_directory(
        r'/home/pavithran/StonePaperScissor/dataset_split/test',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical')

print("Class Name",x_train.class_indices)

steps_per_epoch=2834//batch_size
validation_steps=808//batch_size


model = model_vgg16(img_size,num_classes,epoch)

filepath = '/home/pavithran/StonePaperScissor/weights' + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit_generator(
        x_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epoch,
        validation_data=x_test,
        callbacks=callbacks_list,
        validation_steps=validation_steps)

model.save('model4c.h5')



