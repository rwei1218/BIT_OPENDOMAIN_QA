# BIT OPEN-DOMAIN QA

## Introduction
* Here is our group's project for CCF & Baidu 2019 Reading Comprehension Competition. We apply the competition models to build a open domain QA system, which can process the query-related docs from search engine and output short and informative answers. You can ask any questions you want here. 
* This project provide you the training scripts for LIC 2019, a demo server and the pretrained models. Weather you want to study the reading comprehension problem in DuReader dataset or build a open domain QA engine, you can find what you want here.


## Get Started
1. Download [model files](https://drive.google.com/open?id=1EsRZjUDlXRifYOjZhfjdhQYPHyuPE5dN):`chinese_L-12_H-768_A-12`, `mrc_model` and `rerank_model`. Then unzip and move to`checkpoints`directory. Finally the project structure will be:
```bash
.
├── checkpoints
│   ├── chinese_L-12_H-768_A-12
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   ├── mrc_model
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   └── rerank_model
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
...
```
2. Change config file: `config.py`
3. Run the server: `python server.py`

## TODO LIST
- [ ] Deploy two bert model by tensorflow serving
