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
        self.dropout=nn.Dropout(p=0.3)

    def forward(self,x):
        embeddings=self.dropout(self.backbone(x))
        logits=self.classifier(embeddings)
        return embeddings,logits
    


#loss fn
'''This function takes in {support:embeddings,labels ; query:embeddings,labels} as input, processDs
    it, and returns the loss. 'n' such lossDs are calculated, summed and then optimized as a
    bunch, which forms a task. This entire fn is an episodic, stateless function '''

def MetaIO_params(support_data,support_labels,query_data,query_labels,learner,lossFn,adpSteps):
    sD=support_data
    sL=support_labels
    qD=query_data
    qL=query_labels
    
    #1. Learner adaptation                 
    for step in range(adpSteps):                   #The initial predictions for all the suuport points are accumulated in this param, used to calculate
        _,support_pred=learner(sD)                   #loss, and the model is updated. At the end of this loop, you get the clone model trained on support
        support_loss=lossFn(support_pred,sL)       #data
        learner.adapt(support_loss,allow_unused=True,allow_nograd=True) 
        
    #2. Query loss,accuracy estimaton
    _,query_pred=learner(qD)
    query_loss=lossFn(query_pred,qL)
    query_acc=(query_pred.argmax(dim=1)==qL).sum().item()/qL.size(0)

    return query_loss,query_acc