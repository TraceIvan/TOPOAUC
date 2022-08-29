from sklearn.utils.extmath import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

INT_MAX=pow(2,31)-100000000

class GAUCLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 num_nodes,
                 adj_maxtrix,
                 global_effect_matrix,
                 global_perclass_mean_effect_matrix,
                 train_mask,
                 device,
                 weight_sub_dim=64,
                 weight_inter_dim=64,
                 weight_global_dim=64,
                 beta=0.5,
                 gamma=1,
                 is_ner_weight=True,
                 loss_type='square'):
            
        '''
        loss_type：ExpGAUC，HingeGAUC，SqGAUC
        '''
        super().__init__()

        self.num_classes = num_classes
        self.gamma = gamma
        self.beta=beta
        self.device=device
        self.loss_type=loss_type
        self.is_ner_weight=is_ner_weight


        self.num_nodes=num_nodes
        self.weight_sub_dim=weight_sub_dim
        self.weight_inter_dim=weight_inter_dim
        self.weight_global_dim=weight_global_dim
        self.adj_maxtrix=adj_maxtrix
        self.global_effect_matrix=global_effect_matrix[:,train_mask].to(self.device)
        self.global_perclass_mean_effect_matrix=global_perclass_mean_effect_matrix#[N,C]

        self.I=torch.eye(self.num_nodes,dtype=torch.bool).to(self.device)
        self.adj_self_matrix=adj_maxtrix^torch.diag_embed(torch.diag(adj_maxtrix.int())).bool()|self.I

        #self.weight_sub_matrix = nn.Parameter(torch.Tensor(self.num_nodes, self.weight_sub_dim))
        #self.weight_inter_matrix = nn.Parameter(torch.Tensor(self.num_nodes, self.weight_inter_dim))
        #self.weight_global_matrix = nn.Parameter(torch.Tensor(self.num_nodes, self.weight_global_dim))
        
        self.linear_sub = nn.Linear(self.global_effect_matrix.shape[1],self.weight_sub_dim , bias=False).to(self.device)
        # self.linear_sub_out = nn.Linear(self.weight_sub_dim,1, bias=True).to(self.device)
        self.linear_inter = nn.Linear(self.global_effect_matrix.shape[1],self.weight_inter_dim , bias=False).to(self.device)
        # self.linear_inter_out = nn.Linear(self.weight_inter_dim,1 , bias=True).to(self.device)
        self.linear_global = nn.Linear(self.global_effect_matrix.shape[1],self.weight_global_dim , bias=False).to(self.device)
        # self.linear_global_out = nn.Linear(self.weight_global_dim,1 , bias=True).to(self.device)
    
        # nn.init.normal_(self.linear_sub.weight)
        # nn.init.normal_(self.linear_sub.bias)
        # nn.init.normal_(self.linear_inter.weight)
        # nn.init.normal_(self.linear_inter.bias)
        # nn.init.normal_(self.linear_global.weight)
        # nn.init.normal_(self.linear_global.bias)
        nn.init.uniform_(self.linear_sub.weight, a=0.0, b=1.0)
        # nn.init.normal_(self.linear_sub.bias)
        # nn.init.normal_(self.linear_inter.weight)
        nn.init.uniform_(self.linear_inter.weight, a=0.0, b=1.0)
        # nn.init.normal_(self.linear_inter.bias)
        # nn.init.normal_(self.linear_global.weight)
        nn.init.uniform_(self.linear_global.weight, a=0.0, b=1.0)

        # self.weight_effects_matrix=nn.Parameter(torch.Tensor(self.num_nodes, self.num_classes)).to(self.device)
        # nn.init.normal_(self.weight_effects_matrix)
        # nn.init.uniform_(self.weight_effects_matrix, a=0.0, b=1.0)

    

    def forward(self, pred, target,mask,w_values_dict):

        """
        Args:

        - `pred`: the predicted score vector => (batch_size, num_classes)
        - `target`: the GT label vector => (batch_size, )

        Returns:

        -loss : the loss value

        """

        Y = torch.stack(
            [target.eq(i).float() for i in range(self.num_classes)],
            1).squeeze()

        N = Y.sum(0)  # [classes, 1]
        loss = torch.Tensor([0.]).to(self.device)

        self.global_sub=self.linear_sub(self.global_effect_matrix).sum(dim=-1)
        self.global_inter=self.linear_inter(self.global_effect_matrix).sum(dim=-1)
        self.global_global=self.linear_global(self.global_effect_matrix).sum(dim=-1)

        # self.weighted_effects=self.global_perclass_mean_effect_matrix.cuda()*self.weight_effects_matrix

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i!=j:
                    i_pred_pos=pred[Y[:,i].bool(),:][:,i]
                    i_pred_neg=pred[Y[:,j].bool(),:][:,i]
                    i_pred_pos_expand=i_pred_pos.unsqueeze(1).expand(i_pred_pos.shape[0],i_pred_neg.shape[0])
                    i_pred_pos_sub_neg=i_pred_pos_expand-i_pred_neg

                    if self.loss_type=='SqGAUC':
                        ij_loss=torch.pow(self.gamma-i_pred_pos_sub_neg,2)
                    elif self.loss_type=='HingeGAUC':
                        ij_loss=F.relu(self.gamma-i_pred_pos_sub_neg)
                    elif self.loss_type=='ExpGAUC':
                        ij_loss=torch.exp(-self.gamma * i_pred_pos_sub_neg)
                    
                    if self.is_ner_weight==0:
                        ij_loss=(1/(N[i]*N[j])*ij_loss).sum()
                        loss+=ij_loss
                        continue

                    i_pred_pos_index=torch.nonzero(Y[:,i]).view(-1)
                    i_pred_neg_index=torch.nonzero(Y[:,j]).view(-1)

                    i_pred_pos_adj=self.adj_maxtrix[mask][i_pred_pos_index]
                    i_pred_neg_adj=self.adj_maxtrix[mask][i_pred_neg_index]
                    i_pred_neg_self_adj=self.adj_self_matrix[mask][i_pred_neg_index]

                    i_pred_pos_adj_expand=i_pred_pos_adj.unsqueeze(1).expand(i_pred_pos_adj.shape[0],i_pred_neg_adj.shape[0],i_pred_pos_adj.shape[1])

                    sub_ner=(i_pred_pos_adj_expand^(i_pred_pos_adj_expand&i_pred_neg_self_adj))
                    inter_ner=(i_pred_pos_adj_expand&i_pred_neg_adj)

                    # i_effects=self.weighted_effects[:,i]
                    # j_effects=self.weighted_effects[:,j]
                    # other_effects=torch.sum(self.weighted_effects,dim=1)-j_effects-i_effects

                    max_dim_0=int(np.floor(INT_MAX/sub_ner.shape[2]/sub_ner.shape[1]))
                    if max_dim_0>sub_ner.shape[0]:
                        sub_ner_nonzeros=sub_ner.nonzero(as_tuple=True)
                    else:
                        cal_times=int(np.floor(sub_ner.shape[0]/max_dim_0))
                        if cal_times*max_dim_0<sub_ner.shape[0]:
                            cal_times+=1
                        sub_ner_nonzeros=list(sub_ner[:max_dim_0].nonzero(as_tuple=True))
                        for i in range(1,cal_times):
                            tmp=list(sub_ner[i*max_dim_0:(i+1)*max_dim_0].nonzero(as_tuple=True))
                            for j in range(3):
                                sub_ner_nonzeros[j]=torch.cat((sub_ner_nonzeros[j],tmp[j]))
                    # sub_ner_nonzeros=sub_ner.cpu().nonzero(as_tuple=True)
                    I_sub=torch.cat([sub_ner_nonzeros[0],sub_ner_nonzeros[1]],0).reshape(2,-1)
                    V_sub=self.global_sub[sub_ner_nonzeros[2]]
                    # V_sub=i_effects[sub_ner_nonzeros[2]]
                    S_sub=torch.sparse_coo_tensor(I_sub,V_sub,sub_ner.shape[:-1]).coalesce()
                    vi_sub=S_sub.to_dense()
                    # vi_sub=torch.matmul(sub_ner.float(),self.global_sub).sum(dim=-1)

                    max_dim_0=int(np.floor(INT_MAX/inter_ner.shape[2]/inter_ner.shape[1]))
                    if max_dim_0>inter_ner.shape[0]:
                        inter_ner_nonzeros=inter_ner.nonzero(as_tuple=True)
                    else:
                        cal_times=int(np.floor(inter_ner.shape[0]/max_dim_0))
                        if cal_times*max_dim_0<inter_ner.shape[0]:
                            cal_times+=1
                        inter_ner_nonzeros=list(inter_ner[:max_dim_0].nonzero(as_tuple=True))
                        for i in range(1,cal_times):
                            tmp=list(inter_ner[i*max_dim_0:(i+1)*max_dim_0].nonzero(as_tuple=True))
                            for j in range(3):
                                inter_ner_nonzeros[j]=torch.cat((inter_ner_nonzeros[j],tmp[j]))
                    # inter_ner_nonzeros=inter_ner.cpu().nonzero(as_tuple=True)
                    I_inter=torch.cat([inter_ner_nonzeros[0],inter_ner_nonzeros[1]],0).reshape(2,-1)
                    V_inter=self.global_inter[inter_ner_nonzeros[2]]
                    # V_inter=other_effects[inter_ner_nonzeros[2]]
                    S_inter=torch.sparse_coo_tensor(I_inter,V_inter,inter_ner.shape[:-1]).coalesce()
                    vi_inter=S_inter.to_dense()
                    # vi_inter=torch.matmul(inter_ner.float(),self.global_inter).sum(dim=-1)

                    vl_i=torch.sigmoid((1+vi_sub)/(1+vi_inter))

                    i_nonzeros=i_pred_pos_adj.nonzero(as_tuple=True)
                    I_yi=i_nonzeros[0].reshape(1,-1)
                    V_yi=self.global_global[i_nonzeros[1]]
                    S_yi=torch.sparse_coo_tensor(I_yi,V_yi,i_pred_pos_adj.shape[:-1]).coalesce()
                    vi_g=S_yi.to_dense()
                    # vi_g=torch.matmul(i_pred_pos_adj.float(),self.global_global).sum(dim=-1)

                    non_i_nonzeros=i_pred_neg_adj.nonzero(as_tuple=True)
                    I_non_yi=non_i_nonzeros[0].reshape(1,-1)
                    V_non_yi=self.global_global[non_i_nonzeros[1]]
                    S_non_yi=torch.sparse_coo_tensor(I_non_yi,V_non_yi,i_pred_neg_adj.shape[:-1]).coalesce()
                    v_non_i_g=S_non_yi.to_dense()
                    # v_non_i_g=torch.matmul(i_pred_neg_adj.float(),self.global_global).sum(dim=-1)
                    vg_i=torch.sigmoid(vi_g.unsqueeze(1).expand(vi_g.shape[0],v_non_i_g.shape[0])-v_non_i_g)
                    #v_i=1-(self.beta*vl_i+(1-self.beta)*vg_i)
                    v_i=1-vl_i
                    
                    ij_loss=(1/(N[i]*N[j])*v_i*ij_loss).sum()

                    cur_len=inter_ner.nonzero().shape[0]
                    pos_idx=mask.nonzero().view(-1)[inter_ner.nonzero()[:,0]].numpy()
                    neg_idx=mask.nonzero().view(-1)[inter_ner.nonzero()[:,1]].numpy()
                    values=v_i[inter_ner.nonzero()[:,0],inter_ner.nonzero()[:,1]].cpu().detach().numpy()
                    for cur_pos in range(cur_len):
                        if pos_idx[cur_pos] not in w_values_dict.keys():
                            w_values_dict[pos_idx[cur_pos]]={}
                        if neg_idx[cur_pos] not in w_values_dict[pos_idx[cur_pos]].keys():
                            w_values_dict[pos_idx[cur_pos]][neg_idx[cur_pos]]=[]
                        w_values_dict[pos_idx[cur_pos]][neg_idx[cur_pos]].append(values[cur_pos])

                    loss+=ij_loss
                
        return loss
