import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nets.cyclegan import generator
from utils.utils import (cvtColor, postprocess_output, preprocess_input,
                         resize_image, show_config)


#---------------------------------------------#
#   注意修改model_path、channel和input_shape
#   需要和训练时的设置一样。
#---------------------------------------------#
class CYCLEGAN(object):
    _defaults = {
        #-----------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #-----------------------------------------------#
        "model_path"        : 'model_data/Generator_A2B_horse2zebra.h5',
        #-----------------------------------------------#
        #   输入图像大小的设置
        #-----------------------------------------------#
        "input_shape"       : [256, 256],
        #-------------------------------#
        #   是否进行不失真的resize
        #-------------------------------#
        "letterbox_image"   : True,
    }

    #---------------------------------------------------#
    #   初始化CYCLEGAN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
            self._defaults[name] = value 
        self.generate()
        
        show_config(**self._defaults)
        
    #---------------------------------------------------#
    #   创建生成模型
    #---------------------------------------------------#
    def generate(self):
        self.net = generator(self.input_shape)
        self.net.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

    #---------------------------------------------------#
    #   生成1x1的图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   获得高宽
        #---------------------------------------------------#
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        
        #---------------------------------------------------#
        #   图片传入网络进行预测
        #---------------------------------------------------#
        pr = self.net.predict(image_data)[0]
        
        #--------------------------------------#
        #   将灰条部分截取掉
        #--------------------------------------#
        if nw is not None:
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            
        #---------------------------------------------------#
        #   进行图片的resize
        #---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            
        image = postprocess_output(pr)
        image = Image.fromarray(np.uint8(image))

        return image
