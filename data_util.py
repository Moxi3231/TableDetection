import os
import json
import struct
import numpy as np

import xml.etree.ElementTree as ET

from glob import glob, iglob
from random import shuffle

class DataWalker:
    
    @classmethod
    def __init__(self,relative_paths:list,
                image_root_paths:list = [],
                marmot_index:list = [],
                table_bank = ["",False],
                wild_bank = ["",False],
                batch_size:int = 8) -> None:
        self.absolute_paths_xml = [os.path.abspath(r_path) for r_path in relative_paths]
        
        if image_root_paths == None:
            self.image_root_paths = self.absolute_paths_xml
        else:
            self.image_root_paths = image_root_paths
        self.marmot_index = marmot_index
        self.batch_size = batch_size
        self.table_bank = table_bank
        self.wild_bank = wild_bank

    @staticmethod
    def get_xy_min_wh(cord,label:int = -1):
        """
        Arguments:
        ----------
            Label: 
                -1: Not Defined
                0: Table
                1: Cell
            Cord:
                expects a list of [x,y] coordinates
        
        Returns:
        --------
            [XMin,YMin,Width,Height,Label of the rectangle]
        """
        cord = [list(map(int,xy.split(','))) for xy in cord]

        area_cord = [[(xy[0] + 17) * (xy[1] + 17),i] for i,xy in enumerate(cord)]
        area_cord.sort()
        min_pindex,max_pindex = area_cord[0][1],area_cord[-1][1]
        xy = cord [min_pindex]
        w,h = cord[max_pindex][0] - xy[0],cord[max_pindex][1] - xy[1]
        
        return np.asarray([max(0,xy[0]),max(0,xy[1]),max(0,w),max(0,h),label])

    @classmethod
    def make_sense_xml_marmot(self,file_path:str):
        root = ET.parse(file_path).getroot()
        
        coords = []
        px0, py1, px1, py0 = list(map(lambda x: struct.unpack('!d', bytes.fromhex(x))[0], root.get("CropBox").split()))
        pw = abs(px1 - px0)
        ph = abs(py1 - py0)
        for table in root.findall(".//Composite[@Label='TableBody']"):
            #XMIN YMIN XMAX YMAX FORMAT
            x0p, y0m, x1p,y1m  = list(map(lambda x: struct.unpack('!d', bytes.fromhex(x))[0], table.get("BBox").split()))
            x0 = (x0p - px0)/pw
            x1 = (x1p - px0)/pw
            y0 = (py1 - y0m)/ph
            y1 = (py1 - y1m)/ph
            #Returning XMin YMin Width Height
            coords.append(np.asarray([x0,y0,x1 - x0,y1 - y0,0]))

        return np.asarray(coords)

    @classmethod
    def make_sense_xml_wild(self,file_path:str):
        root = ET.parse(file_path).getroot()
        tdict = dict()
        for bbox in root.findall("object"):
            coords = [int(float(x.text)) for x in bbox.find('bndbox')[:4]]
            tbid = bbox.find('bndbox')[-1].text
            oval = tdict.setdefault(tbid,coords)
            nval = [min(coords[0],oval[0]), min(coords[1],oval[1]), max(coords[2],oval[2]), max(coords[3],oval[3])]
            tdict[tbid] = nval
        
        tb_coords = []
        for tbid in tdict:
            xmin,ymin,xmax,ymax = tdict[tbid]
            tb_coords.append(np.asarray([xmin,ymin,(xmax - xmin),(ymax - ymin),0]))
        if len(tb_coords) == 0:
            return None,None
        return np.asarray(tb_coords),root.find('filename').text

    @classmethod
    def make_sense_xml(self,file_path:str):
        """
        Arguments:
        ----------
        file_path: XML file from which data is to be processsed

        Returns:
        --------
        List of coordinates for tables and cells with their labels and
        image_name.

        (list_of_coordinates,labels,image_file_name)
        """
        root = ET.parse(file_path).getroot()
        image_file_name = root.attrib['filename']
        temp_points = []
        fg_table = False
        for table in root:
            table_cord = table[0].attrib['points'].split()
            x,y,w,h,l = self.get_xy_min_wh(table_cord,0)
            
            if w <= 250  or h <= 250:
                return None,None
            wh_ratio = w/h
            if wh_ratio < 0.3 or wh_ratio > 4.5:
                return None,None
            temp_points.append([x,y,w,h,l])
            fg_table = True
            #for cell in table[1:]:
            #    cell_cord = cell[0].attrib['points'].split()
            #    x,y,w,h,l = self.get_xy_min_wh(cell_cord,1)
            #    if w == 0.0 or h == 0.0:
            #        continue
            #    temp_points.append([x,y,w,h,l])
        if not fg_table:
            return None,None
        
        #shuffle(temp_points)
        temp_points = np.asarray(temp_points)
        #temp_points = np.reshape(temp_points,(,4))
        
        return temp_points,image_file_name

    @classmethod
    def process_all_dirs(self):
        """
        Returns labels and list of directory for image

        """
        labels,image_paths = [],[]
        norms_flag = []
        if self.wild_bank[-1]:
            path = os.path.join(self.wild_bank[0],"xml")
            for pth in iglob(os.path.join(path,"*.xml")):
                boxes,image_name = self.make_sense_xml_wild(pth)
                if image_name == None:
                    continue
                labels.append(boxes)
                image_paths.append(os.path.join(self.wild_bank[0],"images",image_name))
                norms_flag.append(False)

        if self.table_bank[-1]:
            path = self.table_bank[0]
            with open(os.path.join(path,"annotations",self.table_bank[1]),'r') as jfp:
                raw_data = json.load(jfp)
                data = dict()
                for img_obj in raw_data['images']:
                    data[img_obj['id']] = [img_obj['file_name']]
                for annotate_obj in raw_data['annotations']:
                    bbox = annotate_obj['bbox']
                    #Table Label
                    bbox.append(0)
                    data[annotate_obj['image_id']].append(np.asarray(bbox))
                
                for ky in data.keys():
                    row_obj = data[ky]
                    image_paths.append(os.path.join(self.table_bank[0],"images",row_obj[0]))
                    labels.append(np.asarray(row_obj[1:]))
                    norms_flag.append(False)
                
        for i in range(len(self.absolute_paths_xml)):
            if i in self.marmot_index:
                tpaths = [os.path.join(self.absolute_paths_xml[i],"English"),os.path.join(self.absolute_paths_xml[i],"Chinese")]
                
                for path in tpaths:
                    for xml_file_path in iglob(os.path.join(path,"Positive","Labeled","*.xml")):
                        boxes = self.make_sense_xml_marmot(xml_file_path)
                        if len(boxes) == 0:
                            continue
                        labels.append(boxes)
                        norms_flag.append(True)
                        image_paths.append(os.path.join(path,"Positive","Raw",os.path.basename(xml_file_path)[:-4] + ".bmp"))
                    
                    for image_file_path in iglob(os.path.join(path,"Negative","Raw","*.bmp")):
                        boxes = np.asarray([np.asarray([0,0,0,0,-1])])
                        labels.append(boxes)
                        norms_flag.append(True)
                        image_paths.append(image_file_path)
                    
                #return labels,image_paths,norms_flag
                continue

            for xml_file_path in iglob(os.path.join(self.absolute_paths_xml[i],"*.xml")):
                coords,image_file_name = self.make_sense_xml(xml_file_path)
                if coords is None or image_file_name is None:
                    continue
                image_file_path = os.path.join(self.absolute_paths_xml[i],image_file_name)
                labels.append(coords)
                norms_flag.append(False)
                image_paths.append(image_file_path)
        
        return labels,image_paths,norms_flag

    