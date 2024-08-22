# Installation

## mlx-example

```
# https
git clone https://github.com/ml-explore/mlx-examples.git

# ssh
gt clone git@github.com:ml-explore/mlx-examples.git
```

## Data

```
# shenzhi-wang/Llama3-8B-Chinese-Chat
git clone https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat


# taide/Llama3-TAIDE-LX-8B-Chat-Alpha1
# 這比較特別，要先過認證
git clone https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1

# meta-llama/Meta-Llama-3-8B-Instruct
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

```

## qlora

```
git clone https://github.com/artidoro/qlora.git
```

## Environment

`pip install -r requirement.txt`

+ 微調
```
pip install transformers==4.31.0
```

## Docker
```
docker run -it --gpus all --name CCG-DataAnnotation -v "$PWD":/usr/src/app -w /usr/src/app --network host python:3.10

# ctrl + p + q

docker exec -it CCG-DataAnnotation /bin/bash
```

+ eval
```
pip install  git+https://github.com/huggingface/transformers.git
```
