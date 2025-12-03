import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

from qpth.qp import QPFunction


#Backbone ResNet CNN
'''This modified ResNet_18 architectue is used as the backbone for protoMAML. Takes in 224x224 res input and produces as 
    512b 1D embedding tensor. You can play with adding additional conv layers to increase learnt features, or anything
    else'''
class ResNet18_mod(nn.Module):
    def __init__(self,embedding_size,output_classes):
        super().__init__()
        self.backbone=models.resnet18(pretrained=False)
        self.backbone.fc=nn.Identity()                                        # type: ignore #Removes the softmax classification head and gives us 512b GAP embeddings
        self.classifier=nn.Linear(embedding_size,output_classes,bias=False)   #Average embeddings(Ck matrix) layer, set bias=True for including bias in FC Layer
        for name,param in self.named_parameters():
            if not ("layer3" in name or "layer4" in name or "fc" in name or "classifier" in name):     #only final 2 layers are unfreezed and used for training
                param.requires_grad=False
    def forward(self,x):
        embeddings=self.backbone(x)
        logits=self.classifier(embeddings)
        return embeddings,logits
    


#loss fn
"""This loss function is the core of MetaOptNet. Essentially, instead of directly using the embeddings and solving the QP,
    we instead project it to a randomized higher fixed dimension space using Random Fourier Features. This saves us the 
    Hilbert Space (rbf kernel) overhead, converting memory complexity from O(n^2) to O(mn)."""

class RFF(nn.Module):
    
    def __init__(self,in_features: int,out_features:int,sigma:float=1.0):

        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.sigma=sigma

        W=torch.randn(in_features,out_features)/sigma
        b=2*math.pi*torch.rand(out_features)

        self.register_buffer("W",W)
        self.register_buffer("b",b)
        self.scale=math.sqrt(2.0/out_features)

    def forward(self,x:torch.Tensor)->torch.Tensor:                                                                   
        projection=x@self.W+self.b                    #Project input tensor x and apply cosine transformation
        return self.scale*torch.cos(projection)


def metaoptnet_lossfn(
    support_data:torch.Tensor,
    support_labels:torch.Tensor,
    query_data:torch.Tensor,
    query_labels:torch.Tensor,
    learner:nn.Module,
    lossFn:nn.Module,
    rff_dim:int=1024,
    sigma:float=10.0,
    C_reg:float=1.0,
    jitter:float=1e-5
):
    device=support_data.device
    stable_dtype=torch.float64

    # Get embeddings
    support_embeddings,_=learner(support_data)
    query_embeddings,_=learner(query_data)

    # RFF projection
    rff=RFF(in_features=support_embeddings.size(-1),out_features=rff_dim,sigma=sigma)
    rff=rff.to(device=device,dtype=stable_dtype)
    support_features=rff(support_embeddings.to(stable_dtype))
    query_features=rff(query_embeddings.to(stable_dtype))
    n_support=support_features.size(0)
    y_bin=(support_labels==support_labels.min()).double()*2-1

    K=support_features@support_features.T  #Gram matrix
    Q=(y_bin[:,None]*y_bin[None,:])*K
    p=-torch.ones(n_support,device=device,dtype=stable_dtype)
    G=torch.cat([torch.eye(n_support),-torch.eye(n_support)],dim=0).to(device=device,dtype=stable_dtype)
    h=torch.cat([C_reg*torch.ones(n_support),torch.zeros(n_support)],dim=0).to(device=device,dtype=stable_dtype)
    A=y_bin[None,:].to(device=device,dtype=stable_dtype)
    b=torch.zeros(1,device=device,dtype=stable_dtype)

    # Solve QP
    alpha=QPFunction(verbose=False)(Q,p,G,h,A,b)

    # Compute classifier weights
    w=(alpha*y_bin[:,None]).T@support_features  # shape [1, rff_dim]
    logits=(query_features @ w.T).to(torch.float32)  # shape [n_query, 1]
    logits=torch.cat([-logits,logits],dim=1)       # shape [n_query, 2] for cross-entropy

    query_loss=lossFn(logits,query_labels)
    with torch.no_grad():
        query_pred=torch.argmax(logits,dim=1)
        query_acc=(query_pred==query_labels).float().mean().item()

    return query_loss,query_acc