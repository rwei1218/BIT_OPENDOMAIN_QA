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
3. Run the server: `python server.py --config_path=config_v1.json --port=7891`
4. Run the server on gpu: `CUDA_VISIBLE_DEVICES=0 nohup python server.py --config_path=config_v1.json --port=7891`
5. Example for open domain QA (GET & POST): `127.0.0.1:7891/demo/open_domain?query=西红柿炒蛋的做法？`
6. Example for doc based QA (only POST): 
    * `127.0.0.1:7891/demo/doc_based`
    * ```{"querys": ["滑轮原材料是什么？","你们滑轮的是全自动生产吗？","你们滑轮多少钱？"], "doc": "原材料：基体采用高强度挤压铝合金，优质锌合金，超强防腐不锈钢及优质碳素结构钢，部分零件采用性能优良的聚酰胺、聚甲醛等工程塑料。表面处理：表面进行电镀或喷涂等防腐处理，防腐实验轻易超过行业标准的中性盐雾防腐要求72H。结构：独特的结构设计，在满足市场大多数型材需求时，能够实现自动化/半自动化生产。技术服务：我们针对不同区域的需求提供多方面的技术服务。价格：目前根据产品类别不同，价格从低到高有划分不同档次的产品，普通类型的单滑轮3.8元/件，双滑轮7.6元/件。"}```

## TODO LIST
- [ ] Deploy two bert model by tensorflow serving
