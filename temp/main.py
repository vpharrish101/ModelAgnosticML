import torch 
import torch.nn as nn
import torch.nn.utils
import matplotlib.pyplot as plt
import learn2learn as l2l
'''In learn2learn==0.2.0, sometimes the .utils package won't be exported properly, even though it exists in the root folder(atleast in my case). This
   was resolved by editing the __init__.py on the main folder, and editing the "from utils import *" =>"from . import utils" and restarting the kernel'''

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from global_var import TuningParameters
from episode_sample import EpisodeSampler
from learner import ResNet18_mod,MetaIO_params
import test


def main():
    global model,p,e

    p=TuningParameters()
    e=EpisodeSampler(p.train_ways,p.train_samples,p.test_samples)
    device=torch.device('cuda')
    model=ResNet18_mod(p.embedding_size,p.train_ways).to(device)
    lossFn=nn.CrossEntropyLoss()
    learner=l2l.algorithms.MAML(model,lr=0.01,first_order=True)
    optimizer=optim.Adam(learner.parameters(),lr=0.001,weight_decay=1e-4)
    scheduler=CosineAnnealingWarmRestarts(optimizer,T_0=100,T_mult=2)

    train_losses=[]
    train_accuracies=[]
    val_losses=[]
    val_accuracies=[]
    epochs=[]

    for x in range(p.outer_loop):
        total_loss=0.0
        total_acc=0.0
        for y in range(p.inner_loop):
            (sdata,slabels),(qdata,qlabels)=e.sample_train_episode()
            support_data=sdata.to(device)
            support_labels=slabels.to(device)
            query_data=qdata.to(device)
            query_labels=qlabels.to(device)
            learner_clone=learner.clone()
            query_loss,query_acc=MetaIO_params(support_data,support_labels,query_data,query_labels,learner_clone,lossFn,p.adp_steps)
            (query_loss/p.inner_loop).backward(retain_graph=False) #change to True if first_order=False
            total_loss+=query_loss.item()
            total_acc+=query_acc
            print(f"Inner loop {y+1}")
        torch.nn.utils.clip_grad_norm_(learner.parameters(),max_norm=1.0) #type:ignore
        optimizer.step()
        optimizer.zero_grad()
        e._train_episode=None
    

        avg_train_loss = total_loss / p.inner_loop
        avg_train_acc = total_acc / p.inner_loop
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        epochs.append(x)

        # Validation
        val_loss=0.0
        val_acc=0.0
        for _ in range(p.inner_loop):
            (sdata,slabels),(qdata,qlabels)=e.sample_validation_episode()
            support_data=sdata.to(device)
            support_labels=slabels.to(device)
            query_data=qdata.to(device)
            query_labels=qlabels.to(device)
            learner_clone=learner.clone()
            query_loss,query_acc=MetaIO_params(support_data, support_labels, query_data, query_labels, learner_clone, lossFn,p.adp_steps)
            val_loss += query_loss.item()
            val_acc += query_acc
        avg_val_loss = val_loss / p.inner_loop
        avg_val_acc = val_acc / p.inner_loop
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)

        # Plotting
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        clear_output(wait=True)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(epochs, train_accuracies, label='Train Acc')
        plt.plot(epochs, val_accuracies, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"Outer Epoch {x+1}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}")

if __name__=="__main__":
    main()