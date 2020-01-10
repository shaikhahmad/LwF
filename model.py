import torch

# torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import time
import copy

import torchvision.models as models


def MultiClassCrossEntropy(logits, labels, T):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = Variable(labels).cuda()
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    labels = torch.softmax(labels / T, dim=1)
    # print('outputs: ', outputs)
    # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    # print('OUT: ', outputs)
    return Variable(outputs).cuda()

def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


class Model(nn.Module):
    def __init__(self, classes):
        # Hyper Parameters
        self.lr_dec_factor = 10

        self.pretrained = False
        self.momentum = 0.9
        self.weight_decay = 0.0001
        # Constant to provide numerical stability while normalizing
        self.epsilon = 1e-16

        # Network architecture
        super(Model, self).__init__()
        self.model = models.resnet34(pretrained=self.pretrained)
        self.model.apply(kaiming_normal_init)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, classes, bias=False)
        self.fc = self.model.fc
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor = nn.DataParallel(self.feature_extractor)

        # n_classes is incremented before processing new data in an iteration
        # n_known is set to n_classes after all data for an iteration has been processed
        self.n_classes = 0
        self.n_known = 0

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def increment_classes(self, new_classes):
        """Add n classes in the final fc layer"""
        n = len(new_classes)
        print('new classes: ', n)
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_features = self.fc.in_features
        out_features = self.fc.out_features

        if self.n_known == 0:
            new_out_features = n
        else:
            new_out_features = out_features + n
        print('new out features: ', new_out_features)
        self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
        self.fc = self.model.fc

        kaiming_normal_init(self.fc.weight)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n



        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def classify(self, images):
        """
        Classify images by softmax
        Args:
            x: input image batch
            Returns:
                preds: Tensor of size (batch_size,)
         """
        _, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)
        return preds

    def update(self, loader, class_map, init_lr, batch_size, num_epochs):

        self.compute_means = True

        # Save a copy to compute distillation outputs
        prev_model = copy.deepcopy(self)
        # prev_model.cuda()

        try:
            classes = list(set(loader.dataset.targets))
        except:
            classes = list(set(loader.dataset.labels))
        # print("Classes: ", classes)
        print('Known: ', self.n_known)
        if self.n_classes == 1 and self.n_known == 0:
            new_classes = [classes[i] for i in range(1, len(classes))]
        else:
            new_classes = [cl for cl in classes if class_map[cl] >= self.n_known]

        if len(new_classes) > 0:
            self.increment_classes(new_classes)
            self.cuda()

        print("Batch Size (for n_classes classes) : ", len(loader.dataset))
        optimizer = optim.SGD(self.parameters(), lr=init_lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)

        # with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):

            for i, (images, labels) in enumerate(tqdm(loader)):
                seen_labels = []
                images = Variable(torch.FloatTensor(images)).cuda()
                seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
                labels = Variable(seen_labels).cuda()

                optimizer.zero_grad()
                logits = self.forward(images)
                cls_loss = nn.CrossEntropyLoss()(logits, labels)
                dist_loss = 0
                if self.n_classes // len(new_classes) > 1:
                    dist_target = prev_model.forward(images)
                    logits_dist = logits[:, :-(self.n_classes - self.n_known)]
                    dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, T=2)

                loss = dist_loss + cls_loss

                loss.backward()
                optimizer.step()

            print('Epoch [{}/{}], Loss: {}'.format(epoch + 1, num_epochs, i + 1, loss.data))
        self.eval()
        self.n_known = self.n_classes
