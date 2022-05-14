import tensorflow as tf

def convert_to_x_y(boxes):
    """
    Args:
    -----
        Coordinates of boxes in the form Xmin,Ymin and Xmax,Ymax    
    Returns:
    -------
        Returns coordinates of boxes in the form Xcenter,Ycenter,width,height
    """
    
    return tf.concat([(boxes[...,:2] + boxes[...,2:])/2.0, boxes[...,2:]-boxes[...,:2]],axis = -1)

def convert_to_xy_center_from_xy_min_wh(boxes):
    """
    Args:
    ----
        Coordinates of boxes in the form Xmin,Ymin and width, height
    
    Returns:
    --------
        Coordinates in the form XCenter, YCenter and width and height
    """
    return tf.concat([boxes[:,:2] + (boxes[:,2:] / 2.0),boxes[:,2:] ],axis = -1)


def convert_to_min_max_corner(boxes):
    """
    Args:
    -----
        Coordinates of boxes in the form Xcenter,Ycenter,width,height
    
    Returns:
    -------
        Returns coordinates of boxes in the form Xmin,Ymin and Xmax,Ymax
    """
    return tf.concat([boxes[...,:2] - boxes[...,2:]/2.0,boxes[...,:2]+boxes[...,2:]/2.0],axis = -1)

def convert_to_display_format(boxes):
    """
    Args:
    -----
        Coordinates of boxes in the form Xmin,Ymin and Xmax,Ymax

    Returns:
    --------
        Coordinates of the boxes in the form Xmin,Ymin and width, height
    """
    return tf.concat([boxes[...,:2] - (boxes[...,2:] / 2.0), boxes[...,2:] ],axis = -1)

def compute_intersection_over_union(boxes1,boxes2):
    
    boxes1_corners = convert_to_min_max_corner(boxes1)
    boxes2_corners = convert_to_min_max_corner(boxes2) 
    
    left_upper_coordinate = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    right_down_coordinate = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    
    intersection = tf.maximum(0.0, right_down_coordinate - left_upper_coordinate)
    intersection_area = intersection[:,:,0] * intersection[:,:,1]

    boxes1_area = boxes1[:,2] * boxes1[:,3]
    boxes2_area = boxes2[:,2] * boxes2[:,3]
    
    union_area = tf.maximum(boxes1_area[:,None] + boxes2_area - intersection_area, 1e-8)
    
    return tf.clip_by_value(intersection_area/union_area,0.0,1.0)




class AnchorBox:
    """
    Generates Anchor Boxes.
    """

    def __init__(self):
        #Original ratio were 0.5, 1.0, 2.0
        self.aspect_ratios = [0.5,1.0,2.0,3.5]                                                  # For Orientation
        self.scales = [pow(2, x) for x in [0]]                                    # For Different size of anchor box
        self._num_anchors = len(self.aspect_ratios)*len(self.scales)                        # Number of anchors
        self._strides = [pow(2, i) for i in range(3,8)]                                     # Strides in the image
        #Area: 
        #   for cells: ---
        #   for table:
        # ==> self._areas = [pow(a,2) for a in [150.0,200.0,350.0,450.0,500.0]]
        self._areas = [pow(a,2) for a in [75.0,175.0,300.0,450.0,550.0]]
        self._anchors_dimensions = self._compute_dimensions()                               # Dimension of the anchor box generated using the above configuration

    
    def _compute_dimensions(self) -> list:
        """
        Computes dimension: height and width of anchors box for each level
        in given areas [8,32,64,128,...]
        """
        anchor_dimensions = []
        # Anchor with particular area
        for area in self._areas:
            temp_dim = []
            # Differing the orientation of anchors
            for ratio in self.aspect_ratios:
                current_anchor_height = tf.math.sqrt(area/ratio)
                current_anchot_width = area / current_anchor_height
                dims = tf.reshape(tf.stack([current_anchot_width, current_anchor_height], axis=-1), [1, 1, 2])
                # Differing the anchor's size
                for scl in self.scales:
                    temp_dim.append(scl*dims)
            anchor_dimensions.append(tf.stack(temp_dim,axis=-2))
        return anchor_dimensions

    def _get_anchors_util(self, feature_height, feature_width, level):
        """
        Generates Anchors Box:
            Will return anchor boxes for particular feature height, feature width and (level:area)
        """
        # adding 0.5, as center for pixel would x.5
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1)*self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        # centers => represents number of pixels, more specifically the center of pixel

        dims = tf.tile(
            self._anchors_dimensions[level-3], 
            [feature_height, feature_width, 1, 1]
            )

        # dims => height and width
        anchors = tf.concat([centers, dims], axis=-1)
        # finally concatenate both
        return tf.reshape(anchors, [feature_height*feature_width*self._num_anchors, 4])

    def get_anchors(self, image_height, image_width):
        """
        Generate Anchor for given image
        Args:
        -----
            image_height: Height of the input image
            image_width : Width of the input image

        Returns:
            anchor boxes for all the feature maps
            Shape (total_anchors,4)
            total_anchors = image_height * image_width * 9
            9 ===> len(aspect_ratio)*len(scale)
        """
        anchors = [self._get_anchors_util(  
                                            tf.math.ceil(image_height/pow(2, i)), 
                                            tf.math.ceil(image_width/pow(2, i)), 
                                            i) 
                                            
                                        for i in range(3,8)]
        return tf.concat(anchors, axis=0)
