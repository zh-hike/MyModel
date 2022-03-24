# 聚类模型
此次试验为了试验各种损失的性能，

## Loss
### [EAMC](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a2db99fd-55e8-4336-af78-a94c405a9e41/Zhou_和_Shen_-_2020_-_End-to-End_Adversarial-Attention_Network_for_Multi.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220322%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220322T130351Z&X-Amz-Expires=86400&X-Amz-Signature=bb8bee12e03a8b078ab465b73ced7a864ffdc71b9fbb92153c187ff8f0a5c180&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Zhou%2520%25E5%2592%258C%2520Shen%2520-%25202020%2520-%2520End-to-End%2520Adversarial-Attention%2520Network%2520for%2520Multi.pdf%22&x-id=GetObject)

`L_r`, `L_reg`, `L_att`, `L_Dsim`, `L_Dsc`,

### [Completer](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a0b40e65-924c-45af-a65c-e9360f54d144/Lin_等%E3%80%82_-_2021_-_COMPLETER_Incomplete_Multi-view_Clustering_via_Co.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220322%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220322T130449Z&X-Amz-Expires=86400&X-Amz-Signature=e9f7a76d39a9582b8cb46d6315ae4ded0b60e99c73c23caf754194632bc9c511&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Lin%2520%25E7%25AD%2589%25E3%2580%2582%2520-%25202021%2520-%2520COMPLETER%2520Incomplete%2520Multi-view%2520Clustering%2520via%2520Co.pdf%22&x-id=GetObject)
`L_cl`

### [Graph Contrastive Cluster](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/69f9dc99-16bc-4140-af73-275c1d78adc2/Zhong_等%E3%80%82_-_Graph_Contrastive_Clustering.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220322%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220322T130705Z&X-Amz-Expires=86400&X-Amz-Signature=285028670f7c82611db831f309179310cb78981ab7d3f24fe2d326517dc30f76&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Zhong%2520%25E7%25AD%2589%25E3%2580%2582%2520-%2520Graph%2520Contrastive%2520Clustering.pdf%22&x-id=GetObject)
`L_agc`

## 消融实验
### voc

| L_r | L_reg | L_att | L_Dsim | L_Dsc |  acc   |  nmi   |
|:---:|:-----:|:-----:|:------:|:-----:|:------:|:------:|
|  +  |       |       |        |       |  25%   |  22%   |
|  +  |   +   |       |        |       | 57.68% | 54.97% |
|  +  |       |       |        |   +   | 43.64% | 54.30% |
|  +  |       |       |   +    |       | 48.98% | 51.14% |
|  +  |   +   |   +   |        |       | 57.67% | 54.88% |
|  +  |   +   |   +   |   +    |       | 57.50% | 57.76% |
|  +  |   +   |   +   |   +    |   +   | 57.34% | 57.21% |



