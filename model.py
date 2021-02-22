import tensorflow as tf
import matplotlib.pyplot as plt
import dataset
import visualisation

def lenet_5(model_input):
    model_input = tf.reshape(model_input,shape=[-1,model_input[0].shape[0],model_input[0].shape[1],1])        
    conv1 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 5, strides = 1, padding='same',activation='relu')(model_input)    
    maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='valid')(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size = 5,strides=1,padding='same',activation='relu')(maxpool1)
    maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=None,padding='valid')(conv2)
    conv3 = tf.keras.layers.Conv2D(filters=120, kernel_size=5,strides=1,padding='same',activation='relu')(maxpool2)

    full_connection1 = tf.keras.layers.Dense(240,activation='relu')(conv3)
    full_connection2 = tf.keras.layers.Dense(120,activation='relu')(full_connection1)    
    full_connection3 = tf.keras.layers.Dense(4,activation='linear')(full_connection2)

    output = full_connection3   
    return output[:,0,0,:]

if __name__ == "__main__":
	(train_data,train_labels),(test_data,test_labels) = dataset.load_data()    
	print(lenet_5(train_data))

