import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt




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
        
        # Getting the first 5 images
        images_to_pred = self.image[:5]
        
        # Predicting the images
        pred = self.model.predict(images_to_pred)
        
        # Plotting images
        for i in range(images_to_pred.shape[-1]):
            self.display([self.image[i], self.label[i], pred[i]])
            
            
            


