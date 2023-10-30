import torch.nn as nn
from torchvision import models

EMBEDDINGS_SIZE = 512 #1000
### Pretrained Resnet18/34/50  

def load_resnet(num_classes, create_fc=False, model_type=50, pretrained=False, backbone_freeze=None ):
    # assert model_type in [18, 34, 50, 101, 152, 'wideres50']
    if model_type.lower() == "resnet18":
        model_ft = models.resnet18(pretrained=pretrained)
    elif model_type.lower() == "resnet34":
        model_ft = models.resnet34(pretrained=pretrained)
    elif model_type.lower() == "resnet50":
        model_ft = models.resnet50(pretrained=pretrained)
    elif model_type.lower() == "resnet101":
        model_ft = models.resnet101(pretrained=pretrained)
    elif model_type.lower() == "resnet152":
        model_ft = models.resnet152(pretrained=pretrained)
    elif model_type.lower() == "wideresnet50":
        model_ft = models.wide_resnet50_2(pretrained=pretrained)
    elif model_type.lower() == "wideresnet101":
        model_ft = models.wide_resnet101_2(pretrained=pretrained)
    else:
        raise Exception("Invalid resnet type")

    if backbone_freeze:
        model_ft = freeze(model_ft, backbone_freeze)

    if create_fc:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = FC_layer(num_ftrs, num_classes)
    else:
        return {"backbone": nn.Sequential(*(list(model_ft.children())[:-2])), "num_ftrs": model_ft.fc.in_features}
    return model_ft

def freeze(model, freeze_at):
        childrens = [i for i,_ in model.named_children()]
        assert freeze_at in childrens, f"Expected {freeze_at} to be in model.named_children()"
        assert freeze_at != "fc", "Cannot freeze fc layer."

        for i,j in model.named_children():
            for _, param in j.named_parameters():
                param.requires_grad = False
            if i == freeze_at:
                break;
        return model

class FC_layer(nn.Module):
    def __init__(self, num_ftrs, num_classes):
        super(FC_layer, self).__init__()
        self.fc0 = nn.Linear(num_ftrs, 512)
        self.rlu = nn.ReLU()
        self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc0(x)
        x = self.rlu(x)
        # return self.fc(x)
        return self.softmax(self.fc(x))


class simclr_model(nn.Module):
    def __init__(self, model_type, num_classes):
        super(simclr_model, self).__init__()
        bb_net = load_resnet(num_classes, create_fc=False, model_type=model_type)
        self.bb = bb_net["backbone"]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(bb_net["num_ftrs"], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x = self.bb(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return self.softmax(x)

def embed_model(model_type, embedSize, pretrained=True, freeze_at=None):
    # exec(f"model={type}_pyt({embedSize}); return model")
    if model_type.lower() == "resnet18":
        return ResNet18_pyt(embedSize, pretrained, freeze_at=freeze_at)
    
    elif model_type.lower() == "resnet32":
        return ResNet34_pyt(embedSize, pretrained, freeze_at=freeze_at)
    
    elif model_type.lower() == "resnet50":
        return ResNet50_pyt(embedSize, pretrained, freeze_at=freeze_at)
        
    elif model_type.lower() == "wideresnet50":
        return WideResNet50_pyt(embedSize, pretrained, freeze_at=freeze_at)
    else:
        raise Exception("Unknown model type")

class ResNet34_pyt(nn.Module):
    '''
    Last fully connected layer changed to ouput EMBEDDINGS_SIZE-dim vector.
    '''
    
    def __init__(self, embedSize=EMBEDDINGS_SIZE, pretrained=False, freeze_at=None):
        super(ResNet34_pyt, self).__init__()
        model = models.resnet34(pretrained)
        if freeze_at:
            model = freeze(model, freeze_at)
        self.model = model
        self.model.fc = nn.Linear(in_features=512, out_features=embedSize, bias=True)

    def forward(self, image):
        features = self.model(image)
        return features

class ResNet18_pyt(nn.Module):
    '''
    Last fully connected layer changed to ouput EMBEDDINGS_SIZE-dim vector.
    '''

    def __init__(self, embedSize=EMBEDDINGS_SIZE, pretrained=False, freeze_at=None):
        super(ResNet18_pyt, self).__init__()
        model = models.resnet18(pretrained)
        if freeze_at:
            model = freeze(model, freeze_at)
        self.model = model

        self.model.fc = nn.Linear(in_features=512, out_features=embedSize, bias=True)

    def forward(self, image):
        features = self.model(image)
        return features

class ResNet50_pyt(nn.Module):
    '''
    Last fully connected layer changed to ouput EMBEDDINGS_SIZE-dim vector.
    '''

    def __init__(self, embedSize=EMBEDDINGS_SIZE, pretrained=False, freeze_at=None):
        super(ResNet50_pyt, self).__init__()
        model = models.resnet50(pretrained)
        if freeze_at:
            model = freeze(model, freeze_at)
        self.model = model

        #self.model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        self.model.fc = nn.Linear(in_features=2048, out_features=embedSize, bias=True)

    def forward(self, image):
        features = self.model(image)
        return features


class WideResNet50_pyt(nn.Module):
    '''
    Last fully connected layer changed to ouput EMBEDDINGS_SIZE-dim vector.
    '''

    def __init__(self, embedSize=EMBEDDINGS_SIZE, pretrained=False, freeze_at=None):
        super(WideResNet50_pyt, self).__init__()
        model = models.wide_resnet50_2(pretrained)
        if freeze_at:
            model = freeze(model, freeze_at)
        self.model = model

        #self.model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        self.model.fc = nn.Linear(in_features=2048, out_features=embedSize, bias=True)
    
    def forward(self, image):
        features = self.model(image)
        return features
