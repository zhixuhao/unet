from keras import backend as K

def universal_dice_coef_multilabel(numLabels):
    def dice_coef(y_true, y_pred):
        dice = 0
        for index in range(numLabels):
            dice += dice_coef_calcucate(y_true[:, :, :, index], y_pred[:, :, :, index])
        return dice / numLabels  # taking average


    return dice_coef


def universal_dice_coef_loss(numLabels):
    def dice_coef_loss(y_true, y_pred):
        dice = 0
        for index in range(numLabels):
            dice += dice_coef_calcucate(y_true[:, :, :, index], y_pred[:, :, :, index])
        return 1 - dice / numLabels  # taking average
    
    return dice_coef_loss

def dice_coef_calcucate(y_true, y_pred):
    smooth = 0.0001
    y_true_f = K.flatten(y_true) #(K.mean(y_true, axis = 0))
    y_pred_f = K.flatten(y_pred) #(K.mean(y_pred, axis = 0))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss_calcucate(y_true, y_pred):
    return 1 - dice_coef_calcucate(y_true, y_pred)
    
'''    
def universal_dice_coef_multilabel_vectors(numLabels):
    def dice_coef(y_true, y_pred):
        dice = 0
        for index in range(numLabels):
            dice += dice_coef_vectors_calculate(y_true[:, :, :, index], y_pred[:, :, :, index])
        return dice / numLabels  # taking average


    return dice_coef

def universal_dice_coef_loss_vectors(numLabels):
    def dice_coef_loss_vectors(y_true, y_pred):
        dice = 0
        for index in range(numLabels):
            dice += dice_coef_vectors_calculate(y_true[:, :, :, index], y_pred[:, :, :, index])
        return 1 - K.mean(dice)/ numLabels # taking average
    
    return dice_coef_loss_vectors
    
def dice_coef_vectors_calculate(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.dot(y_true, K.transpose(y_pred))
    union = K.dot(y_true,K.transpose(y_true)) + K.dot(y_pred,K.transpose(y_pred))
    return (2. * intersection + smooth) / (union + smooth)
'''
