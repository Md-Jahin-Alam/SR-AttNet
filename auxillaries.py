import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

# -----------------------------------------------------------------------------
# Dataset Augments

def adjust_data(img,mask):  
    mask_thresh = 128/256
    img = img / 255.
    mask = mask / 255.
    mask[mask > mask_thresh] = 1.0
    mask[mask <= mask_thresh] = 0.0        
    return (img, mask)  #Input and Output image conditioning

augments = ImageDataGenerator(rotation_range=0.1,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest')


# -----------------------------------------------------------------------------
# Losses and Metrics

def precision(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1))) 
    precision = true_positives / (predicted_positives + K.epsilon()) 
    return precision 

def recall(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) 
    recall = true_positives / (possible_positives + K.epsilon()) 
    return recall 

def dice_coef(y_true, y_pred):
    smooth=1
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    
    return (2.0 * intersection + smooth) / (union + smooth)

def dice_coef_Thresh(y_true, y_pred):
    y_pred_thresh = tf.cast(y_pred + 0.5, tf.int32)
    y_pred_thresh = tf.cast(y_pred_thresh, tf.float32)
    smooth=1
    y_true = K.flatten(y_true)
    y_pred_thresh = K.flatten(y_pred_thresh)
    intersection = K.sum(y_true * y_pred_thresh)
    union = K.sum(y_true) + K.sum(y_pred_thresh)
    
    return (2.0 * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred)

def iou_Thresh(y_true, y_pred): #Jaccard Index
    y_pred = tf.cast(y_pred + 0.5, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    smooth=1
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def fbeta_score(y_true, y_pred, beta=1.0): 
    if beta < 0: 
        raise ValueError('The lowest choosable beta is zero (only precision).') 
    if K.sum(K.round(K.clip(y_true, 0.0, 1.0))) == 0: 
        return 0.0 
    p = precision(y_true, y_pred) 
    r = recall(y_true, y_pred) 
    bb = beta*beta
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon()) 
    return fbeta_score 

def fbeta2(y_true, y_pred): 
    return fbeta_score(y_true, y_pred, beta=2.0)


# metrics=[ precision, 
#           recall, 
#           dice_coef_Thresh, 
#           fbeta2, 
#           iou_Thresh, 
#           tf.keras.metrics.AUC(num_thresholds=11, curve='PR', from_logits=True)
#         ]
                        