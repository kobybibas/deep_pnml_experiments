from dataset_utilities import create_cifar10_dataloaders
from dataset_utilities import create_cifar10_random_label_dataloaders
from dataset_utilities import create_mnist_dataloaders
from dataset_utilities import create_svhn_dataloaders
from mpl import Net
from resnet import resnet20, load_pretrained_resnet20_cifar10_model
from wide_resnet import WideResNet


class Experiment:
    def __init__(self, type, params):
        if type not in ['pnml_cifar10',
                        'random_labels',
                        'out_of_dist_svhn',
                        'out_of_dist_noise',
                        'pnml_mnist']:
            raise NameError('No experiment type: %s' % type)
        self.params = params
        self.exp_type = type
        self.executed_get_params = False

    def get_params(self):
        if self.exp_type == 'pnml_cifar10':
            self.params = self.params['nml_vanilla']
        elif self.exp_type == 'random_labels':
            self.params = self.params['random_labels']
        elif self.exp_type == 'out_of_dist_svhn':
            self.params = self.params['nml_vanilla']
        elif self.exp_type == 'out_of_dist_noise':
            self.params = self.params['nml_vanilla']
        elif self.exp_type == 'pnml_mnist':
            self.params = self.params['nml_mnist']
        else:
            raise NameError('No experiment type: %s' % self.exp_type)

        self.executed_get_params = True
        return self.params

    def get_dataloaders(self, data_folder='./data'):

        if self.executed_get_params is False:
            _ = self.get_params()

        if self.exp_type == 'pnml_cifar10':
            trainloader, testloader, classes = create_cifar10_dataloaders(data_folder,
                                                                          self.params['batch_size'],
                                                                          self.params['num_workers'])
            dataloaders = {'train': trainloader,
                           'test': testloader,
                           'classes': classes}
        elif self.exp_type == 'random_labels':
            trainloader, testloader, classes = create_cifar10_random_label_dataloaders(data_folder,
                                                                                       self.params['batch_size'],
                                                                                       self.params['num_workers'],
                                                                                       label_corrupt_prob=self.params[
                                                                                           'label_corrupt_prob'])
            dataloaders = {'train': trainloader,
                           'test': testloader,
                           'classes': classes}
        elif self.exp_type == 'out_of_dist_svhn':
            trainloader, testloader, classes = create_cifar10_dataloaders(data_folder,
                                                                          self.params['batch_size'],
                                                                          self.params['num_workers'])
            _, testloader_svhn, classes_svhn = create_svhn_dataloaders(data_folder,
                                                                       self.params['batch_size'],
                                                                       self.params['num_workers'])
            dataloaders = {'train': trainloader,
                           'test': testloader_svhn,
                           'classes': classes,
                           'classes_svhn': classes_svhn}

        elif self.exp_type == 'out_of_dist_noise':
            trainloader, testloader, classes = create_cifar10_dataloaders(data_folder,
                                                                          self.params['batch_size'],
                                                                          self.params['num_workers'])
            classes_noise = ('Noise',) * 10
            dataloaders = {'train': trainloader,
                           'test': testloader,
                           'classes': classes,
                           'classes_noise': classes_noise}

        elif self.exp_type == 'pnml_mnist':
            trainloader, testloader, classes = create_mnist_dataloaders(data_folder,
                                                                        self.params['batch_size'],
                                                                        self.params['num_workers'])
            dataloaders = {'train': trainloader,
                           'test': testloader,
                           'classes': classes}

        else:
            dataloaders = {}
            raise NameError('No experiment type: %s' % self.exp_type)

        return dataloaders

    def get_model(self):

        if self.exp_type == 'pnml_cifar10':
            model = load_pretrained_resnet20_cifar10_model(resnet20())
        elif self.exp_type == 'random_labels':
            model = WideResNet()
        elif self.exp_type == 'out_of_dist_svhn':
            model = load_pretrained_resnet20_cifar10_model(resnet20())
        elif self.exp_type == 'out_of_dist_noise':
            model = load_pretrained_resnet20_cifar10_model(resnet20())
        elif self.exp_type == 'pnml_mnist':
            model = Net()
        else:
            raise NameError('No experiment type: %s' % self.exp_type)

        return model

    def get_exp_name(self):

        if self.exp_type == 'pnml_cifar10':
            name = 'pnml_cifar10'
        elif self.exp_type == 'random_labels':
            name = 'random_labels'
        elif self.exp_type == 'out_of_dist_svhn':
            name = 'out_of_dist_svhn'
        elif self.exp_type == 'out_of_dist_noise':
            name = 'out_of_dist_noise'
        elif self.exp_type == 'pnml_mnist':
            name = 'pnml_mnist'
        else:
            raise NameError('No experiment type: %s' % self.exp_type)

        return name
