import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#Backbone ResNet CNN
'''This modified ResNet_18 architectue is used as the backbone for protoMAML. Takes in 224x224 res input and produces as 
    512b 1D embedding tensor. You can play with adding additional conv layers to increase learnt features, or anything
    else'''
class ResNet18_mod(nn.Module):
    
    def __init__(self,embedding_size,output_classes):
        super().__init__()
        self.backbone=models.resnet18(pretrained=True)
        self.backbone.fc=nn.Identity()                                        #Removes the softmax classification head and gives us 512b GAP embeddings #type:ignore
        self.classifier=nn.Linear(embedding_size,output_classes,bias=False)   #Average embeddings(Ck matrix) layer, set bias=True for including bias in FC Layer
        for name,param in self.named_parameters():
            if not ("layer3" in name or "layer4" in name or "fc" in name or "classifier" in name):     #only final 2 layers are unfreezed and used for training
                param.requires_grad=False

    def forward(self,x):
        embeddings=self.backbone(x)
        logits=self.classifier(embeddings)
        return embeddings,logits
    

'''This function takes in {support:embeddings,labels ; query:embeddings,labels} as input, processes
   it, and returns the loss. 'n' such losses are calculated, summed and then optimized as a
   bunch, which forms a task. This entire fn is an episodic, stateless function'''

def protoMAML_lossfn(support_data,support_labels,query_data,query_labels,learner,lossFn,adpSteps):
    
    def Ck(support_embeddings,support_labels):
        classes=torch.unique(support_labels)
        mean_embed=[]
        for c in classes:
            class_embeddings=support_embeddings[support_labels==c]
            prototype=class_embeddings.mean(dim=0)
            mean_embed.append(prototype)
        return torch.stack(mean_embed)
    
    support_embeddings,_=learner(support_data)
    mean_embed=Ck(support_embeddings,support_labels)
    prototype_init_loss=F.mse_loss(learner.module.classifier.weight,mean_embed)
    learner.adapt(prototype_init_loss,allow_unused=True,allow_nograd=True)

    for step in range(adpSteps):
        _,support_logits=learner(support_data)
        support_loss=lossFn(support_logits,support_labels)
        learner.adapt(support_loss,allow_unused=True,allow_nograd=True)

    _,query_logits=learner(query_data)
    query_loss=lossFn(query_logits,query_labels)
    query_pred=torch.argmax(query_logits,dim=1)
    query_acc=(query_pred==query_labels).float().mean()
    
    return query_loss,query_acc
