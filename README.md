# Source Free Unsupervised Graph Domain Adaptation

This repository is the official implementation of Source Free Unsupervised Graph Domain Adaptation, accepted by WSDM 2024

## Abstract

Graph Neural Networks (GNNs) have achieved great success on a variety of tasks with graph-structural data, among which node classification is an essential one. Unsupervised Graph Domain Adaptation (UGDA) shows its practical value of reducing the labeling cost for node classification. It leverages knowledge from a labeled graph (i.e., source domain) to tackle the same task on another unlabeled graph (i.e., target domain). Most existing UGDA methods heavily rely on the labeled graph in the source domain. They utilize labels from the source domain as the supervision signal and are jointly trained on both the source graph and the target graph. However, in some real-world scenarios, the source graph is inaccessible because of privacy issues. Therefore, we propose a novel scenario named Source Free Unsupervised Graph Domain Adaptation (SFUGDA). In this scenario, the only information we can leverage from the source domain is the well-trained source model, without any exposure to the source graph and its labels. As a result, existing UGDA methods are not feasible anymore. To address the non-trivial adaptation challenges in this practical scenario, we propose a model-agnostic algorithm called SOGA for domain adaptation to fully exploit the discriminative ability of the source model while preserving the consistency of structural proximity on the target graph. We prove the effectiveness of the proposed algorithm both theoretically and empirically. The experimental results on four cross-domain tasks show consistent improvements in the Macro-F1 score and Macro-AUC.

## Training

To train the models (GCN-SOGA by default) in the paper, run this command:

```
python main.py
```

Please change the source dataset and the target dataset manually.

## Hyperparameter settings

For GraphSAGE, GAT, DANE, UDAGCN, we conduct carefully grid search as follows.(In GraphSAGE, the sample2 and sample1 in the tables represents the node sample number of the second and the first layer, respectively.)

| Model     | tuning parameter                        |                                                              |
| --------- | --------------------------------------- | ------------------------------------------------------------ |
| GraphSAGE | neighbor sample2    neighbor sample1    | [5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,150,200,250,300,350,400,450,500] |
| GAT       | attention head                          | [1, 2, 4, 8, 16, 32, 64, 128]                                |
| DANE      | SINGLE（weight on the adversarial loss) | [1, 2, 3, 4, 5]                                              |
| UDAGCN    | path_len                                | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                              |

The best hyperparameter on the macro-F1 score are:

| Model     | tuning parameter                               | DBLP->ACM | ACM->DBLP | ACM-D->ACM-S | ACM-S->ACM-D |
| --------- | ---------------------------------------------- | --------- | --------- | ------------ | ------------ |
| GraphSAGE | [neighbor sample2,    neighbor sample1]        | [40, 35]  | [60, 10]  | [20, 10]     | [5, 25]      |
| GAT       | attention head                                 | 128       | 64        | 128          | 64           |
| DANE      | LAMBDA_SINGLE（weight on the adversarial loss) | 5         | 3         | 5            | 5            |
| UDAGCN    | path_len                                       | 10        | 10        | 5            | 4            |

The best hyperparameter on the macro-AUC score are:

| Model     | tuning parameter                        | DBLP->ACM | ACM->DBLP | ACM-D->ACM-S | ACM-S->ACM-D |
| --------- | --------------------------------------- | --------- | --------- | ------------ | ------------ |
| GraphSAGE | [neighbor sample2,    neighbor sample1] | [20, 70]  | [25, 450] | [15, 5]      | [10, 25]     |
| GAT       | attention head                          | 128       | 64        | 64           | 64           |
| DANE      | SINGLE（weight on the adversarial loss) | 3         | 2         | 5            | 4            |
| UDAGCN    | path_len                                | 8         | 3         | 1            | 2            |


## License

All content in this repository is licensed under the [MIT license](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt).

