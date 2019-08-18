# Deep pNML: Predictive Normalized Maximum Likelihood for Deep Neural Networks
Koby Bibas, Yaniv Fogel and Meir Feder

This is the official implementioation of "Deep pNML: Predictive Normalized Maximum Likelihood for Deep Neural Networks"

https://arxiv.org/abs/1904.12286

## Get started:

1. Clone the repository

2. Intsall requeirement 

```
pip install -r requirements.txt
```

3. Run basic experimnet:

```
CUDA_VISIBLE_DEVICES=0 python src/main.py -t pnml_cifar10
```

## Experimnets:

The experimnet options are:

1. pnml_cifar10: running pNML on CIFAR10 dataset.
2. random_labels: runing pNML on CIFAR10 dataset that its labels are random.
3. out_of_dist_svhn: trainset is CIFAR10. Execute pNML on SVHN dataset.
4. out_of_dist_noise:  trainset is CIFAR10. Execute pNML on Noise images.
5. pnml_mnist: runining pNML on MNIST dataset.
4. pnml_cifar10_lenet: trainset is CIFAR10. Execute pNML with LeNet architecture.

The parameters of each experimnet can be change in the parameters file: src\params.json

Raw results for Nips 2019:
https://drive.google.com/file/d/1PouRaGxw6TNcqPOqUjpQ3G5TXYYhORlW/view?usp=sharing

### Citing
```
@misc{bibas2019deep,
    title={Deep pNML: Predictive Normalized Maximum Likelihood for Deep Neural Networks},
    author={Koby Bibas and Yaniv Fogel and Meir Feder},
    year={2019},
    eprint={1904.12286},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

