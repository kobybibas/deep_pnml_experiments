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

4. Analyze the outputs using jupyter notebooks:

```
├── notebooks
│   ├── adversarial_attack.ipynb
│   ├── distributions_metrics.py
│   ├── mixture_lenet_cifar10.ipynb
│   ├── mixture_out_of_distribution.ipynb
│   ├── mixture_random_labels.ipynb
│   ├── mixture_resnet18_cifar10.ipynb
│   ├── model_selection_exploration.ipynb
│   ├── out_of_distribution.ipynb
│   ├── plot_functions.ipynb
│   ├── pnml_lenet_cifar10.ipynb
│   ├── pnml_resnet18_cifar10.ipynb
│   ├── random_labels.ipynb
│   ├── result_summary.ipynb
│   └── twice_universality.ipynb
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

## Results

Raw results are in:
https://drive.google.com/open?id=1sMCZo2aoei7UxahQONAOf8gLp5Hjb1WQ


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

