# Special Thanks TO : https://github.com/cvdfoundation/fashionpedia?tab=readme-ov-file 


wget https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/visual.onnx

wget https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/textual.onnx

wget https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip

wget s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json

unzip val_test2020.zip

python generate_file_name_and_metadata_json.py