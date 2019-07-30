# Deep pNML: Predictive Normalized Maximum Likelihood for Deep Neural Networks
Koby Bibas, Yaniv Fogel and Meir Feder

This is the official implementioation of "Deep pNML: Predictive Normalized Maximum Likelihood for Deep Neural Networks"

https://arxiv.org/abs/1904.12286

### Abstract
The Predictive Normalized Maximum Likelihood (pNML) scheme has been recently suggested for universal learning in the individual setting, where both the training and test samples are individual data. The goal of universal learning is to compete with a "genie" or reference learner that knows the data values, but is restricted to use a learner from a given model class. The pNML minimizes the associated regret for any possible value of the unknown label. Furthermore, its min-max regret can serve as a pointwise measure of learnability for the specific training and data sample. In this work we examine the pNML and its associated learnability measure for the Deep Neural Network (DNN) model class. As shown, the pNML outperforms the commonly used Empirical Risk Minimization (ERM) approach and provides robustness against adversarial attacks. Together with its learnability measure it can detect out of distribution test examples, be tolerant to noisy labels and serve as a confidence measure for the ERM. Finally, we extend the pNML to a "twice universal" solution, that provides universality for model class selection and generates a learner competing with the best one from all model classes.

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

