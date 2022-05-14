import box_util
import tensorflow as tf

from importlib import reload
from tensorflow import keras

reload(box_util)

AnchorBox = box_util.AnchorBox
compute_intersection_over_union = box_util.compute_intersection_over_union
convert_to_min_max_corner = box_util.convert_to_min_max_corner



class LabelEncoder:
    def __init__(self) -> None:
        #Here anchor box is used to calculate the IoU,
        # Which are ultimately used to 
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2])
        
    def _match_anchor_boxes(self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4):
        iou_matrix = compute_intersection_over_union(anchor_boxes, gt_boxes)
        #tf.print("GT BOXES\n",gt_boxes,"\n")
        mx_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_indexes = tf.argmax(iou_matrix, axis=1)
        
        positive_mask = tf.greater_equal(mx_iou, match_iou)
        negative_mask = tf.less(mx_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))

        return (matched_gt_indexes, tf.cast(positive_mask, dtype=tf.float32), tf.cast(ignore_mask, tf.float32))

    def _compute_box_trg(self, anchor_boxes, matched_gt_boxes):
        """
        Conversion of boundary box
        Mached_gt_boxes: stacked up objects, 
         
        Anchor Box Generated via this class are gathered,
        Offset is calculated and then encoding is done
        More like normaliation, than encoding.
        
        Encoding: for training of real boundary box
        And then normalization.
        """
        box_trg = tf.concat([
            (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
            tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:])
        ], axis=-1)
        box_trg = box_trg / self._box_variance
        return box_trg
    
    def _encode_labels_for_single_image(self, image_shape, gt_boxes, cls_ids):
        
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_index, postive_mask, ignore_mask = self._match_anchor_boxes(anchor_boxes, gt_boxes)
        
        #y,_ = tf.unique(matched_gt_index)
        #y = float(len(y))
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_index)
        #box_trg = self._compute_box_trg(anchor_boxes=anchor_boxes, matched_gt_boxes=matched_gt_boxes)
        box_trg = matched_gt_boxes
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_index)

        cls_trg = tf.where(tf.not_equal(postive_mask, 1.0), -1.0, matched_gt_cls_ids)
        cls_trg = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_trg)
        cls_trg = tf.expand_dims(cls_trg, axis=-1)

        label = tf.concat([box_trg, cls_trg], axis=-1)
        
        #if tf.reduce_sum(postive_mask,axis=-1) == 0.0:
        #    tf.print("NO MATCH FOUND")
        
        #tv1,t2v1 = tf.reduce_sum(tf.where(tf.equal(0.0,cls_ids),1.0,0.0)),tf.reduce_sum(tf.where(tf.equal(0.0,cls_trg),1.0,0.0))
        #tv2,t2v2 = tf.reduce_sum(tf.where(tf.equal(1.0,cls_ids),1.0,0.0)),tf.reduce_sum(tf.where(tf.equal(1.0,cls_trg),1.0,0.0))
        #ylen = len(tf.unique(tf.where(tf.not_equal(postive_mask, 1.0), -1.0, tf.cast(matched_gt_index,dtype=tf.float32)))[0])
        #if tv1 > t2v1 or tv2 > t2v2:
        #if tv2 > t2v2 or int(tv2) > int(ylen):
        #    tf.print("BOX G",gt_boxes)
        #    tf.print("ANCHOR BOX",anchor_boxes)
        #
        #    tf.print("Number of initial matches:",tf.reduce_sum(postive_mask,axis=-1))
        #    tf.print("Stats: Table [O | F]",tv1,t2v1)
        #    tf.print("Stats: CELL  [O | TF | UF]",tv2,t2v2,ylen)
        return label


    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]
        
        labels = tf.TensorArray(
           dtype=tf.float32, size=batch_size, dynamic_size=True)
        
        for i in range(batch_size):
            label = self._encode_labels_for_single_image(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
            
        processed_images = keras.applications.resnet.preprocess_input(batch_images)
        return processed_images, labels.stack()





class BoxDecoder:
    def __init__(
                self,
                num_classes = 2,
                box_variance = [0.1,0.1,0.2,0.2]
                ) -> None:

        self.num_classes = num_classes
       
        self._box_variance = tf.convert_to_tensor(box_variance, dtype=tf.float32)
        self._anchor_box = AnchorBox()
    
    def decode_box_predictions_util(self, anchor_boxes, box_predictions):
        # Decoding | Normalization process
        boxes = box_predictions * self._box_variance
        # Here instead of predicting the box co-ordinate via model,
        # the difference is predicted i.e., once anchor is placed then how far,
        # is it from real anchor. Here model tries to predict the offset.
        boxes = tf.concat(
            [
                (boxes[:,:,:2] * anchor_boxes[:,:,2:]) + anchor_boxes[:,:,:2],
                tf.math.exp(boxes[:,:,2:]) * anchor_boxes[:,:,2:]
            ],axis=-1)

        #boxes = convert_to_min_max_corner(boxes)
        return boxes