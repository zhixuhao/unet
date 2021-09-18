import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras
import numpy as np

'''

    Added by Sohaib Anwaar
    
    sohaibanwaar36@gmail.com

'''

class ValidatePredictions(keras.callbacks.Callback):
    '''
    
        This custom callback helps you to validate your prediction while training.
    
    '''
    def __init__(self,model, generator):
        
        '''
        
            This is the init of this class which takes all the required params to get started
            
            params: model     : (keras.model)      Model which you are using for training
            params: generator : (Image_generator)  Validation or Training Generator
        
        '''
        
        super(ValidatePredictions, self).__init__()

        self.image, self.label = next(generator)
        self.model = model
        
    
    
    
    def display(self, display_list):
        '''
            
            Display Fucntion which display your image, groundtruth and prediction
            
            params: display_list : (list) List of images to display in a sequece
                                    e.g [image1, image2, image3] where image1, image2, image3 is numpy array
                                    
        '''
        
        # Title List to display on the top of the image
        gt_list = ["Image", "Label", "Prediction"]
        
        # Plot figure size
        plt.figure(figsize=(15, 15))
        
        # Appending all plot figs
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(gt_list[i])
            plt.imshow(display_list[i])
            plt.axis('off')
        
        # Showing images
        plt.show()

    def on_epoch_begin(self, epoch, logs=None):
        
        '''
        
            This function will execute on the start of the epoch and plot the prediction
            
            params: epoch : (int) Epoch Number
            params: logs : (dict) logs generated till now, i.e val_loss, loss, accuracy etc you can explore
                                  with logs.keys()
            
        '''
        
        # Clear ouput before plotting
        clear_output(wait=True)
        
        # Getting the first 5 images
        images_to_pred = self.image[:5]
        
        # Predicting the images
        pred = self.model.predict(images_to_pred)
        
        # Plotting images
        print("Start epoch {}".format(epoch))
        for i in range(images_to_pred.shape[-1]):
            self.display([self.image[i], self.label[i], pred[i]])
            
            
            


class TrainingPlot(keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig('output/Epoch-{}.png'.format(epoch))
            plt.close()