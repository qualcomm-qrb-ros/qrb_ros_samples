#/bin/bash
wget -O 1.jpg https://raw.githubusercontent.com/laxmimerit/dog-cat-full-dataset/master/data/train/dogs/dog.12498.jpg
wget -O 2.jpg  -O 1.jpg  https://raw.githubusercontent.com/laxmimerit/dog-cat-full-dataset/master/data/train/dogs/dog.12497.jpg
wget -O 3.jpg  https://raw.githubusercontent.com/laxmimerit/dog-cat-full-dataset/master/data/train/dogs/dog.12491.jpg
wget -O 4.jpg  https://raw.githubusercontent.com/laxmimerit/dog-cat-full-dataset/master/data/train/dogs/dog.12401.jpg

wget https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/visual.onnx
wget https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/textual.onnx