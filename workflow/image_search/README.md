# Image Search Workflow

This project demonstrates a simple yet powerful LLM-powered research agent. This implementation is based directly on the tutorial: [LLM Agents are simply Graph — Tutorial For Dummies](https://zacharyhuang.substack.com/p/llm-agent-internal-as-a-graph-tutorial).

Reference : https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-agent

## Thanks To

Example pictures are from : https://github.com/laxmimerit/dog-cat-full-dataset/tree/master

- https://raw.githubusercontent.com/laxmimerit/dog-cat-full-dataset/master/data/train/dogs/dog.12498.jpg

ONNX models are from CLIP-AS-SERVICE : https://github.com/jina-ai/clip-as-service

- https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/visual.onnx
- https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/textual.onnx

## Features (planing)

- Used original source code (not API) of CLIP preprocess
- Used ONNX GPU to accelerate CLIP Encoding
- Used original numpy to calculate similairies between Image Features and Text Features of CLIP
- Dynamically adding pictures and turing into vectors 
- Search with natural language

## Example Outputs
![ExampleOutputs](./example1.jpg)

## Getting Started

1. Install the packages you need with this simple command:
```bash
pip install -r requirements.txt
```
2. Run agent :

```bash
python test_clip_preprocess_and_image_search.py
```

## How It Works?

TBD