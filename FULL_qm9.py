import math
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader,random_split
import torch.nn.functional as F
from einops import rearrange
import os
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
encode_dir={'O':[0,0,0,0,1,16],'C':[0,0,0,1,0,12],'H':[0,0,1,0,0,1],'N':[0,1,0,0,0,14],'F':[1,0,0,0,0,19]}
dir = './qm9/qm9_raw'
def process_input(dir_path):
    #sum h x y
    all_h = []
    all_x = []
    all_y = []
    all_mask = []
    all_edge = []
    cur_dirs = os.listdir(dir_path)
    for item in tqdm(cur_dirs,desc="process data"):
        if len(item) != 20:
            continue
        cur_dir = dir_path+"//"+item
        cur_y,cur_h,cur_x,cur_mask,cur_edge = read_xyz_file(cur_dir)
        all_y.append(cur_y)
        all_x.append(cur_x)
        all_h.append(cur_h)
        all_mask.append(cur_mask)
        all_edge.append(cur_edge)
    tools = abs(min(all_y)[0])
    for i in range(len(all_y)):
        all_y[i][0] = all_y[i][0] + tools
    return all_h,all_x,all_y,all_mask,all_edge
def read_xyz_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        em_len = int(lines[0])
        p = []
        p.append(float(lines[1].split()[6:7][0])*2.72114)
        # p.append(float(lines[1].split()[6:7][0])) # alpha
        # p.append(float(lines[1].split()[10:11][0]))
        atom_name = []
        atom_coords = []
        node_mask = []
        edge_mask = []
        x_max = float("-inf")
        y_max = float("-inf")
        z_max = float("-inf")
        x_min = float("inf")
        y_min = float("inf")
        z_min = float("inf")
        for i in range(em_len):
            line = lines[i+2].split()
            atom_name.append(encode_dir[line[0]])
            tmp_list = []
            for coord in line[1:4]:
                try:
                    float(coord)
                    tmp_list.append(float(coord))
                except ValueError:
                    tmp = int(coord[-2:])
                    cur_num = float(coord[0:3])*math.pow(10,tmp)
                    #print(coord,cur_num)
                    tmp_list.append(cur_num)
            atom_coords.append(tmp_list)
            node_mask.append(1)
        bone_line = 30-em_len
        x = 0
        y = 0
        z = 0
        for coord in atom_coords:
            x = x + coord[0]
            y = y + coord[1]
            z = z + coord[2]
        x_mean = float(x) / float(em_len)
        y_mean = float(y) / float(em_len)
        z_mean = float(z) / float(em_len)
        tmp_dir=[0,0,0,0,0,0]
        for i in range(bone_line):
            node_mask.append(0)
            atom_name.append(tmp_dir)
            atom_coords.append([x_mean,y_mean,z_mean])
        for i in range(30):
            edge_mask.append(node_mask)
    return p,atom_name,atom_coords,node_mask,edge_mask
class Pin_cloud(data.Dataset):
    def __init__(self, cur_y=None,cur_h=None,cur_x=None,cur_mask=None,cur_edge=None):
        self.cur_h = cur_h
        self.cur_x = cur_x
        self.cur_y = cur_y
        self.cur_mask = cur_mask
        self.cur_edge = cur_edge

    def __getitem__(self, index):
        return (torch.tensor(self.cur_h[index]).float(),
                torch.tensor(self.cur_x[index]).float(),
                torch.tensor(self.cur_y[index]).float(),
                torch.tensor(self.cur_mask[index]).float(),
                torch.tensor(self.cur_edge[index]).float())

    def __len__(self):
        return len(self.cur_x)

class EGNN(nn.Module):
    def __init__(self,fea_dim=4,coord_dim=3,edg_dim=100,out_feat_dim=100):
        super(EGNN, self).__init__()
        self.node_mlp=nn.Sequential(
            nn.Linear(fea_dim+edg_dim, (fea_dim+edg_dim)*2),
            nn.SiLU(),
            nn.Linear((fea_dim+edg_dim)*2, out_feat_dim)
        )
        self.edg_mlp=nn.Sequential(
            nn.Linear(fea_dim*2 + 1, edg_dim),
            nn.SiLU(),
            nn.Linear(edg_dim,edg_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(edg_dim, edg_dim*2),
            nn.Tanh(),
            nn.Linear(edg_dim*2, 1)
        )
        self.aij = nn.Sequential(
            nn.Linear(edg_dim,1),
            nn.Sigmoid()
        )
    def forward(self,h,x,m):
        cur_len=h.shape[-2]
        batch_size=h.shape[0]
        # hi and hj
        c = rearrange(h, 'b i d -> b i () d')
        b = rearrange(h, 'b j d -> b () j d')
        c, b = torch.broadcast_tensors(c, b)
        h_i = rearrange(c, 'a b c d -> a (b c) d')
        h_j = rearrange(b, 'a b c d -> a (b c) d')# batch node_num feat_num
        # xi and xj
        g = rearrange(x, 'b i d -> b i () d')
        f = rearrange(x, 'b j d -> b () j d')
        radial = rearrange(g-f, 'a b c d -> a (b c) d')
        coord_diff = (radial ** 2).sum(dim = -1, keepdim = True)
        m_ij = self.edg_mlp(torch.cat([h_i,h_j,coord_diff],dim=-1)).reshape(batch_size,cur_len,cur_len,-1)
        m_ij = self.aij(m_ij)*m_ij
        m_ij = m_ij * m.unsqueeze(dim=-1)
        # update node
        agg = m_ij.sum(dim = -2)
        h = self.node_mlp(torch.cat([h,agg],dim=-1))
        # update coord
        cur = (self.coord_mlp(m_ij)).reshape(batch_size,-1,1)*radial
        agg = cur.reshape(batch_size,cur_len,cur_len,-1).sum(dim = -2)
        x = x + torch.div(agg,cur_len-1)
        return h,x


class WMEGNN(nn.Module):
    def __init__(self,in_dim=4,coord_dim=3,out_dim=100):
        super(WMEGNN, self).__init__()
        self.egnn1 = EGNN(fea_dim=in_dim, coord_dim=coord_dim, edg_dim=256, out_feat_dim=128)
        self.egnn2 = EGNN(fea_dim=128, coord_dim=coord_dim, edg_dim=256, out_feat_dim=128)#,hid_dim=256)
        self.egnn3 = EGNN(fea_dim=128, coord_dim=coord_dim, edg_dim=256, out_feat_dim=128)#,hid_dim=256)
        self.egnn4 = EGNN(fea_dim=128, coord_dim=coord_dim, edg_dim=256, out_feat_dim=128)
        self.egnn5 = EGNN(fea_dim=128, coord_dim=coord_dim, edg_dim=256, out_feat_dim=128)#,hid_dim=256)
        self.egnn6 = EGNN(fea_dim=128, coord_dim=coord_dim, edg_dim=256, out_feat_dim=128)#,hid_dim=256)
        self.egnn7 = EGNN(fea_dim=128, coord_dim=coord_dim, edg_dim=256, out_feat_dim=128)
        self.egnn8 = EGNN(fea_dim=128, coord_dim=coord_dim, edg_dim=256, out_feat_dim=128)
        self.egnn9 = EGNN(fea_dim=128, coord_dim=coord_dim, edg_dim=256, out_feat_dim=128)
        self.egnn10 = EGNN(fea_dim=128, coord_dim=coord_dim, edg_dim=256, out_feat_dim=128)
        self.egnn11 = EGNN(fea_dim=128, coord_dim=coord_dim, edg_dim=256, out_feat_dim=128)
        self.egnn12 = EGNN(fea_dim=128, coord_dim=coord_dim, edg_dim=256, out_feat_dim=128)
        self.pred = nn.Sequential(
            # nn.Linear(in_features=256, out_features=128),
            # nn.SELU(),
            nn.Linear(in_features=128, out_features=64),
            nn.SELU(),
            nn.Linear(in_features=64, out_features=32),
            nn.SELU(),
            nn.Linear(in_features=32, out_features=1)
        )
        self.embed_0 = nn.Linear(134, 128)
        self.embed_1 = nn.Linear(256,128)
        self.embed_2 = nn.Linear(256, 128)
        self.embed_3 = nn.Linear(256, 128)
        self.embed_4 = nn.Linear(256, 128)
        self.embed_5 = nn.Linear(256, 128)
        self.embed_6 = nn.Linear(256, 128)
        self.embed_7 = nn.Linear(256, 128)
        self.embed_8 = nn.Linear(256, 128)
        self.embed_9 = nn.Linear(256, 128)
        self.embed_10 = nn.Linear(256, 128)
        self.embed_11 = nn.Linear(256, 128)

    def forward(self,h,x,node_mask,edge_mask):
        b = h.shape[0]
        m = edge_mask
        h1, x1 = self.egnn1(h, x,m)
        h1 = self.embed_0(torch.cat([h,h1],dim=-1))
        x1 = (x1 + x)/2
        h2, x2 = self.egnn2(h1, x1,m)
        h2 = self.embed_1(torch.cat([h1,h2],dim=-1))
        x2 = (x2 + x1) / 2
        h3, x3 = self.egnn3(h2,x2,m)
        h3 = self.embed_2(torch.cat([h2,h3],dim=-1))
        x3 = (x3 + x2) / 2

        h4, x4 = self.egnn4(h3, x3,m)
        h4 = self.embed_3(torch.cat([h3,h4],dim=-1))
        x4 = (x4 + x3) / 2
        h5, x5 = self.egnn5(h4, x4,m)
        h5 = self.embed_4(torch.cat([h4,h5],dim=-1))
        x5 = (x5 + x4) / 2
        h6, x6 = self.egnn6(h5, x5,m)
        h6 = self.embed_5(torch.cat([h5,h6],dim=-1))
        x6 = (x6 + x5) / 2

        h7, x7 = self.egnn7(h6, x6, m)
        h7 = self.embed_6(torch.cat([h5,h6],dim=-1))
        x7 = (x7 + x6) / 2
        h8, x8 = self.egnn8(h7, x7, m)
        h8 = self.embed_7(torch.cat([h6,h7],dim=-1))
        x8 = (x8 + x7) / 2
        h9, x9 = self.egnn9(h8, x8, m)
        h9 = self.embed_8(torch.cat([h8,h9],dim=-1))
        x9 = (x9 + x8) / 2

        h9 = h9*(node_mask.unsqueeze(-1))
        h = torch.sum(h9, dim=-2)
        pred = self.pred(h)
        return pred,x9
def train_loop(n_epochs, train_loader,valid_loader,batch_size,optimizer, model, scheduler,loss_f):
    for epoch in tqdm(range(1, n_epochs + 1),desc="train_looping"):
        for t_h_train, t_x_train, t_c_train, t_mask_train, t_edge_train in train_loader:
            if torch.cuda.is_available():
                t_h_train = t_h_train.cuda()
                t_x_train = t_x_train.cuda()
                t_c_train = t_c_train.cuda()
                t_mask_train = t_mask_train.cuda()
                t_edge_train = t_edge_train.cuda()
            pred,coord = model(t_h_train,t_x_train,t_mask_train,t_edge_train)
            coord = coord*(t_mask_train.unsqueeze(-1))
            t_x_train = t_x_train*(t_mask_train.unsqueeze(-1))
            loss_train = loss_f(pred, t_c_train.float())+torch.mean(torch.mean(torch.sum((coord - t_x_train)**2,dim=-1),dim=-1),dim=-1)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch}, Training loss {loss_train:.9f}")
        child = 0
        all_sum_mae = 0
        with torch.no_grad():
            for t_h_train, t_x_train, t_c_train,t_mask_train, t_edge_train  in valid_loader:
                if torch.cuda.is_available():
                    t_h_train = t_h_train.cuda()
                    t_x_train = t_x_train.cuda()
                    t_c_train = t_c_train.cuda()
                    t_mask_train = t_mask_train.cuda()
                    t_edge_train = t_edge_train.cuda()
                pred,coord = model(t_h_train,t_x_train,t_mask_train,t_edge_train)
                child = child + 1
                mae = torch.abs(pred - t_c_train).sum(dim=-1).sum(dim=0)
                mae = (mae).cpu().detach().numpy()
                all_sum_mae = all_sum_mae + float(float(mae) / float(batch_size))
        print(f"cur_valid_res: {(all_sum_mae / child):.9f}", all_sum_mae, child)

def main():
    all_h, all_x, all_y, all_mask,all_edge = process_input(dir)
    pro_data = Pin_cloud(cur_h=all_h,cur_x=all_x,cur_y=all_y,cur_mask=all_mask,cur_edge=all_edge)
    torch.manual_seed(1001)
    train_dataset,test_dataset = random_split(pro_data,[100000,len(pro_data)-100000])
    train_len = 100000
    train_batch_size = 64
    valid_dataset,test_dataset = random_split(test_dataset,[18000,len(test_dataset)-18000])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)#,collate_fn=collate_fn)
    test_batch_size = train_batch_size
    valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size, shuffle=False,
                             drop_last=True)#,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                             drop_last=True)#,collate_fn=collate_fn)
    my_model = WMEGNN(in_dim=6,out_dim=256,coord_dim=3)
    if torch.cuda.is_available():
        my_model = my_model.cuda()
    optimizer = torch.optim.AdamW(my_model.parameters(), lr=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40,T_mult=2, eta_min=1e-9, last_epoch=-1,verbose=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,
                                                    pct_start=0.25,
                                                    div_factor=1e4,
                                                    anneal_strategy='cos',
                                                    total_steps=1000*(train_len//train_batch_size))
    train_loop(
        n_epochs=1000,
        train_loader=train_loader,
        valid_loader=valid_loader,
        batch_size=train_batch_size,
        optimizer=optimizer,
        model=my_model,
        loss_f=nn.L1Loss(),
        scheduler=scheduler
    )

    #test dataset
    with torch.no_grad():
        child = 0
        all_sum_mae = 0
        for t_h_train, t_x_train, t_c_train,t_mask_train,t_edge_train  in tqdm(test_loader, desc="test dataset looping"):
            if torch.cuda.is_available():
                t_h_train = t_h_train.cuda()
                t_x_train = t_x_train.cuda()
                t_c_train = t_c_train.cuda()
                t_mask_train = t_mask_train.cuda()
                t_edge_train = t_edge_train.cuda()
            pred,coord = my_model(t_h_train,t_x_train,t_mask_train,t_edge_train)
            child = child + 1
            mae = torch.abs(pred[:,0] - t_c_train[:,0]).sum(dim=0)
            mae = (mae).cpu().detach().numpy()
            all_sum_mae = all_sum_mae + float(float(mae) / float(train_batch_size))
        print(f"cur_test_res: {(all_sum_mae / child):.9f}", all_sum_mae, child)

if __name__ == '__main__':
    main()
    process_input(dir)
    print(torch.__version__)
    print("Task accomplished!!")
