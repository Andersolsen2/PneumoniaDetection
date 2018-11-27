!/bin/bash\

git clone https://www.github.com/matterport/Mask_RCNN.git\
python setup.py -q install\

git clone https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5\
ls -lh mask_rcnn_coco.h5
