from keras.backend import tensorflow_backend as K

#IOU calc
iou_smooth=1.

def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    return 2*(intersection + iou_smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + iou_smooth)


def IOU_calc_loss(y_true, y_pred):
    return 1-IOU_calc(y_true, y_pred)