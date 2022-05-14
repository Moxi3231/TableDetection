import box_util
import tensorflow as tf
import encoder_decoder

from importlib import reload

reload(box_util)

AnchorBox = box_util.AnchorBox
convert_to_min_max_corner = box_util.convert_to_min_max_corner

##
# GIOU: Generalized Intersection of Union
# tfa.losses.GIoULoss() Tensorflow Addons
##
##
###
#
# CIoU: Complete Intersection Over Union
# Takes three parameter into consideration 
#   1. Intersection Over Union
#   2. Normalized Central Point Distance
#   3. Aspect Ratio

# CIoU = S(B1,B2) + D(B1,B2) + V(B1,B2)
# S: 1 - IoU
# D: Normalized Central Point Distance
# V: Aspect Ratio
class CIoULoss(tf.losses.Loss):
    def __init__(self) -> None:
        super(CIoULoss,self).__init__(reduction = "none",name="CIoULoss")
        self.pi2 = tf.constant(22/7) ** 2
        
    def call(self,y_true,y_pred):
        """
        y_true,y_pred: 
                    (Batch_Size,Num_Anchor_Box,4)  
                    XCenter,YCenter, Width, Height
                        ==> Converted to:
                    xmin,ymin, xmax,ymax
        
        """
        #Intersection Over Union
        y_pred = convert_to_min_max_corner(y_pred)
        y_true = convert_to_min_max_corner(y_true)
        
        box_pred_width_height = tf.maximum(0.0, y_pred[...,2:] - y_pred[...,:2])
        box_true_width_height = tf.maximum(0.0, y_true[...,2:] - y_true[...,:2])

        left_upper_coordinate = tf.maximum(y_true[..., :2], y_pred[..., :2])
        right_down_coordinate = tf.minimum(y_true[..., 2:], y_pred[..., 2:])
        
        intersection = tf.maximum(0.0, right_down_coordinate - left_upper_coordinate)
        intersection_area = intersection[:,:,0] * intersection[:,:,1]
        
        boxes1_area = box_pred_width_height[...,0] * box_pred_width_height[...,1]
        boxes2_area = box_true_width_height[...,0] * box_true_width_height[...,1]
        
        union_area = tf.maximum(boxes1_area + boxes2_area - intersection_area, 1e-8)
            # 1 - iou # Defining Loss ==> iou: 1 then complete, else not complete
            # For IOU 1, box predicted == box_proposed_original
            # Hence loss = 1 - iou
        siou = 1.0 - tf.clip_by_value(intersection_area/union_area,0.0,1.0)
        #Normalized Central Point Distance
        center_point_pred = (y_pred[...,2:] + y_pred[...,:2])/2.0
        center_point_true = (y_true[...,2:] + y_true[...,:2])/2.0

        #Distance D
        dist_d = tf.reduce_sum((center_point_pred - center_point_true) ** 2,axis = -1)
        #Distance C
        dist_c = tf.reduce_sum((left_upper_coordinate - right_down_coordinate) ** 2,axis = -1)
        param_d = tf.math.divide_no_nan(dist_d,dist_c)
        
        #Aspect Ratio
        aratio_width_height_true = tf.math.atan(tf.math.divide_no_nan(box_true_width_height[...,0],box_true_width_height[...,1]))
        aratio_width_height_pred = tf.math.atan(tf.math.divide_no_nan(box_pred_width_height[...,0],box_pred_width_height[...,1]))

        param_v = (aratio_width_height_pred - aratio_width_height_true) ** 2
        param_v = 4 * tf.math.divide_no_nan(param_v,self.pi2)
        
        alpha = tf.where(tf.greater(siou,0.5),tf.math.divide_no_nan(param_v,1 - param_v + siou),0.0)
        
        param_v = alpha * param_v
        return siou + param_d + param_v

class SmoothL1Loss(tf.losses.Loss):
    """
    Implementation of smooth l1 loss
    BOX LOSS
    """
    def __init__(self,delta) -> None:
        super(SmoothL1Loss,self).__init__(reduction="none",name="SmoothL1Loss")
        self._delta = delta

    def call(self,y_true,y_pred):
        diff = y_true - y_pred
        
        abs_diff = tf.abs(diff)
        squared_diff = abs_diff ** 2
        loss = tf.where(tf.less(abs_diff,self._delta),0.5*squared_diff,abs_diff-0.5)
        # for i >> if abs_diff[i] < self._delta then out[i] = 0.5*squared_diff[i] else out[i] = abs_diff[i] - 0.5
        return tf.reduce_sum(loss,axis = -1)

class FocalLoss(tf.losses.Loss):
    """
    Implementation of focal loss
    Classification loss
    """
    def __init__(self,alpha,gamma) -> None:
        super(FocalLoss,self).__init__(reduction = "none",name = "FocalLoss")
        self._alpha = alpha
        self._gamma = gamma

    def call(self,y_true,y_pred):
        #tf.print("YPRED",tf.shape(y_pred),"YTRUE",tf.shape(y_true))
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true,1.0),self._alpha,(1.0-self._alpha))
        pt = tf.where(tf.equal(y_true,1.0),probs,1-probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss,axis = -1)



class CombinedLoss(tf.losses.Loss):
    """
    Combined loss for box regression and class predictions
    """
    def __init__(self,num_classes = 3,alpha = 0.25,gamma = 2.0, delta = 1.0) -> None:
        super(CombinedLoss,self).__init__(reduction = "auto",name = "CombinedLoss")
        self._cls_loss = FocalLoss(alpha=alpha,gamma=gamma)
        #self._cls_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        #self._box_loss = SmoothL1Loss(delta=delta)
        self._box_loss = CIoULoss()
        #In case if object score is supposed to be measured
        self._num_classes = num_classes# - 1

    def call(self,y_true,y_pred):
        y_pred = tf.cast(y_pred,tf.float32)
        box_labels = y_true[:,:,:4]
        box_predictions = y_pred[:,:,:4]

        #Change Here For OBJ SCORE
        cls_labels = tf.one_hot(tf.cast(y_true[:,:,4],dtype = tf.int32),depth = self._num_classes ,dtype=tf.float32)
        cls_labels = tf.cast(cls_labels,dtype=tf.float32)
        
        #cls_labels = y_true[:,:,4]

        #obj_scores = tf.where(tf.greater_equal(y_true[:,:,4],0.0),1.0,0.0)
        #cls_labels = tf.concat([cls_labels,tf.expand_dims(obj_scores,axis = -1)],axis=-1)
        
        cls_predictions = y_pred[:,:,4:]
       
        postive_mask = tf.cast(tf.greater(y_true[:,:,4],-1.0),dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:,:,4],-2.0),dtype=tf.float32)
        
        #tf.print("CL",tf.shape(box_labels),tf.shape(box_predictions))
        #tf.print("BX",tf.shape(cls_labels),tf.shape(cls_predictions))
        
        box_loss = self._box_loss(box_labels, box_predictions)
        cls_loss = self._cls_loss(cls_labels, cls_predictions)
        
        cls_loss = tf.where(tf.equal(ignore_mask,1.0),0.0,cls_loss)
        box_loss = tf.where(tf.equal(postive_mask,1.0),box_loss,0.0)
        normalizer = tf.reduce_sum(postive_mask,axis = -1)

        cls_loss = tf.math.divide_no_nan(tf.reduce_sum(cls_loss,axis = -1),normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss,axis = -1),normalizer)
        
        #tloss = cls_loss + box_loss
        #return tloss
        #tf.print("BOXLOSS:",box_loss,"  CLSLOSS:",cls_loss,"\n")
        return cls_loss + box_loss