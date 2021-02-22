from argparse import ArgumentParser
import tensorflow.compat.v1 as tf1
import tensorflow  as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys 
import model
import dataset
import csv

class classifier:
    
    def __init__(self,ses,lr,train_labels,train_data,test_data,test_labels,batch_size):
        self.ses=ses
        self.accuracy=0
        self.batch_size=batch_size
        self.batch_no=0
        self.train_data=train_data
        self.train_labels=train_labels
        self.test_data=test_data
        self.test_labels=test_labels

        self.input=tf1.placeholder(shape=[None,self.train_data[0].shape[0],self.train_data[0].shape[1]],dtype=tf.uint8,name="input")
        self.predictions=tf1.placeholder(shape=[None,4],dtype=tf.float32,name="predictions")

        self.perceptron=tf.identity(model.lenet_5(self.input/255),name="perceptron")
        loss_fn=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.predictions,logits=self.perceptron))
        self.loss=tf.identity(loss_fn,name="loss")
        self.optimizer=tf1.train.GradientDescentOptimizer(learning_rate=lr,name="optimizer")
        self.train_op=self.optimizer.minimize(self.loss)

        self.acc_val=tf.cast(tf.equal(tf.argmax(self.predictions,1),tf.argmax(self.perceptron,1)),dtype=tf.float32)
        self.acc=tf.reduce_mean(self.acc_val)

        self.ses.run(tf1.global_variables_initializer())

    def predict(self,image):
        return self.ses.run(tf.argmax(self.perceptron,1),feed_dict={self.input:[image]})[-1]

    def train(self,episodes):
        val=0
        loss_val = []
        batch_no=0
        episodes = episodes*440//self.batch_size
        while episodes: 
            batch_data,batch_labels=self.get_next_batch()    
            _,self.accuracy=self.ses.run((self.train_op,self.acc),feed_dict={self.input:batch_data,self.predictions:batch_labels})
           
            val=self.ses.run(self.acc,feed_dict={self.input:self.test_data,self.predictions:self.test_labels})
            loss_val.append(val)
            print("\r accuracy {} % & episodes = {} ".format(val*100,episodes//(440//self.batch_size)),end='')
            sys.stdout.flush()                            
            self.batch_no += self.batch_size
            episodes-=1                

    def plot(self,x,y):
        plt.xlabel('batch')
        plt.ylabel('accuracy')
        plt.title('training graph')
        plt.plot(x,y)
        plt.show()
        plt.close()                  

    def get_next_batch(self):        
        batch_data = self.train_data[self.batch_no:self.batch_no+self.batch_size]
        batch_labels = self.train_labels[self.batch_no:self.batch_no+self.batch_size]
        self.batch_no=(self.batch_no+self.batch_size)%(440 - self.batch_size)
        return batch_data,batch_labels

def show_image(image):
    plt.imshow(image,cmap='gray')
    plt.show()

if __name__ == '__main__':
    parser=ArgumentParser()
    parser.add_argument("--episodes",type=int,dest="episodes",default=3000,help="Number of training episdoes")
    parser.add_argument("--restore_memory",action="store_true",help="Restore model parameters")
    parser.add_argument("--lr",type=float,dest="lr",default=0.02,help="Learning rate")
    parser.add_argument("--batch_size",type=float,dest="batch_size",default=10,help="number of data")
    args=parser.parse_args()
    episodes=args.episodes
    batch_size = args.batch_size
    lr=args.lr
    tf1.disable_eager_execution()
    ses=tf1.Session()
    (train_data,train_label),(test_data,test_label)=dataset.load_data()  

    train_label = tf.keras.backend.eval(tf.one_hot(train_label,depth=4))  
    test_label = tf.keras.backend.eval(tf.one_hot(test_label,depth=4))  

    classifier=classifier(ses,lr,train_label,train_data,test_data,test_label,batch_size)
    classifier.train(episodes)
    os.chdir('./../../')
    with open('output.csv','w') as csvwriter:
        writer = csv.writer(csvwriter)
        for i in range(len(test_data)):
            predictions = classifier.predict(test_data[i])
            print(predictions,np.argmax(test_label[i]))        
            writer.writerow(str(predictions)+str(np.argmax(test_label[i])))
            