# Video face recognition based on RetinaFace and FaceNet

---

## Catalog
1. [Notes](#Notes)
2. [Environment](#Environment)
3. [How2predict](#How2predict)

## Notes
The library contains two networks, retinaface and facenet, both of which use different weights.    
When using the networks, you must pay attention to the choice of weights and the matching of trunk and weights.      

## Environment
pytorch==1.2.0 


## How2predict
1. The project comes with its own backbone for the retinaface model and facenet model of mobilenet. 
2.. In the retinaface.py file, modify the model_path and backbone in the following section to make them correspond to the trained files.  
```python
_defaults = {
    "retinaface_model_path" : 'model_data/Retinaface_mobilenet0.25.pth',
    #-----------------------------------#
    #   Select RetinaFace backone as MobileNet
    #-----------------------------------#
    "retinaface_backbone"   : "mobilenet",
    "confidence"            : 0.5,
    "iou"                   : 0.3,
    #----------------------------------------------------------------------#
    #  If or not the image size limit is needed.
    #  The input image size will affect the FPS significantly, you can reduce the input_shape if you want to speed up the detection speed.
    #  When enabled, it will limit the input image size to input_shape. otherwise, use the original image for prediction.
    #  The input_shape can be adjusted according to the size of the input image, note that it is a multiple of 32, e.g. [640, 640, 3]
    #----------------------------------------------------------------------#
    "retinaface_input_shape": [640, 640, 3],
    #-----------------------------------#
    #   Whether the image size limit is required.
    #-----------------------------------#
    "letterbox_image"       : True,
    
    "facenet_model_path"    : 'facenet_mobilenet0.25.pth',
    #-----------------------------------#
    #   Select FaceNet backbone for MobileNet.
    #-----------------------------------#
    "facenet_backbone"      : "mobilenet",
    "facenet_input_shape"   : [160,160,3],
    "facenet_threhold"      : 0.9,

    "cuda"                  : True
}
```
3. Upload an image in the face_dataset folder. The naming rules of face_dataset are XXX_1.jpg, XXX_2.jpg.
4. Run encoding.py to encode the images inside the face_dataset. The model will generate the corresponding database face encoding data files in the model_data folder.
5. Upload a photo or video that needs to be recognized in the img folder and change the path of the image/video in the predict.py function.
6. Run predict.py.

## Experiment results
https://drive.google.com/drive/folders/1twdFGTBXaLMNAzMBl7wrfn4Ves_tjiwb?usp=share_link

Let's see the results on the image side.  
![ben9](https://user-images.githubusercontent.com/90971979/206825165-6e61d4f0-0c7c-4c17-9b7c-cb411f34513b.jpg)
