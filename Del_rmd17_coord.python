import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader,random_split
import torch.nn.functional as F
from einops import rearrange
from sklearn.preprocessing import OneHotEncoder
dir = './rmd17/rmd17_malonaldehyde.npz'

def process_input(dir_path):
    #sum h x y
    data = np.load(dir)
    all_h = []
    all_x = data['coords'].tolist()
    all_y = data['energies'].tolist()
    ori_h = data['nuclear_charges'].tolist()
    encoder = OneHotEncoder(sparse=False)
    categories = np.array(ori_h).reshape(-1, 1)
    one_hot_encoded = encoder.fit_transform(categories).tolist()
    all_len = data['energies'].shape[0]
    for j in range(len(ori_h)):
        one_hot_encoded[j].append(ori_h[j])
    for i in range(all_len):
        all_h.append(one_hot_encoded)
    tools = abs(min(all_y))
    for i in range(len(all_y)):
        all_y[i] = all_y[i] + tools
    return all_h,all_x,all_y
class Pin_cloud(data.Dataset):
    def __init__(self, cur_y=None,cur_h=None,cur_x=None):
        self.cur_h = cur_h
        self.cur_x = cur_x
        self.cur_y = cur_y

    def __getitem__(self, index):
        return (torch.tensor(self.cur_h[index]).float(),
                torch.tensor(self.cur_x[index]).float(),
                torch.tensor(self.cur_y[index]).float())

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
    def forward(self,h,x):
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
        self.pred = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.SiLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.SiLU(),
            nn.Linear(in_features=32, out_features=1)
        )
        self.embed_0 = nn.Linear(128+in_dim, 128)
        self.embed_1 = nn.Linear(256,128)
        self.embed_2 = nn.Linear(256, 128)
        self.embed_3 = nn.Linear(256, 128)
        self.embed_4 = nn.Linear(256, 128)
        self.embed_5 = nn.Linear(256, 128)
        self.embed_6 = nn.Linear(256, 128)
        self.embed_7 = nn.Linear(256, 128)
        self.embed_8 = nn.Linear(256, 128)

    def forward(self,h,x):
        h1, x1 = self.egnn1(h, x)
        h1 = self.embed_0(torch.cat([h,h1],dim=-1))
        h2, x2 = self.egnn2(h1, x1)
        h2 = self.embed_1(torch.cat([h1,h2],dim=-1))
        h3, x3 = self.egnn3(h2,x2)
        h3 = self.embed_2(torch.cat([h2,h3],dim=-1))

        h4, x4 = self.egnn4(h3, x3)
        h4 = self.embed_3(torch.cat([h3,h4],dim=-1))
        h5, x5 = self.egnn5(h4, x4)
        h5 = self.embed_4(torch.cat([h4,h5],dim=-1))
        h6, x6 = self.egnn6(h5, x5)
        h6 = self.embed_5(torch.cat([h5,h6],dim=-1))

        h7, x7 = self.egnn7(h6, x6)
        h7 = self.embed_6(torch.cat([h5,h6],dim=-1))
        h8, x8 = self.egnn8(h7, x7)
        h8 = self.embed_7(torch.cat([h6,h7],dim=-1))
        h9, x9 = self.egnn9(h8, x8)
        h9 = self.embed_8(torch.cat([h8,h9],dim=-1))

        h = torch.sum(h9, dim=-2)
        pred = self.pred(h)
        return pred
def train_loop(n_epochs, train_loader,valid_loader,batch_size,optimizer, model, scheduler,loss_f):
    for epoch in tqdm(range(1, n_epochs + 1),desc="train_looping"):
        for t_h_train, t_x_train, t_c_train in train_loader:
            t_h_train = t_h_train.float()
            t_x_train = t_x_train.float()
            t_c_train = t_c_train.unsqueeze(-1)
            if torch.cuda.is_available():
                t_h_train = t_h_train.cuda()
                t_x_train = t_x_train.cuda()
                t_c_train = t_c_train.cuda()
            pred = model(t_h_train,t_x_train)
            loss_train = loss_f(pred, t_c_train.float())
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch}, Training loss {loss_train:.9f}")
        with torch.no_grad():
            for t_h_train, t_x_train, t_c_train in valid_loader:
                t_h_train = t_h_train.float()
                t_x_train = t_x_train.float()
                t_c_train = t_c_train.unsqueeze(-1)
                if torch.cuda.is_available():
                    t_h_train = t_h_train.cuda()
                    t_x_train = t_x_train.cuda()
                    t_c_train = t_c_train.cuda()
                pred = model(t_h_train, t_x_train)
                loss_train = loss_f(pred, t_c_train.float())
            print(f"cur_valid_res: {(loss_train):.9f}")

def main():
    all_h, all_x, all_y = process_input(dir)
    pro_data = Pin_cloud(cur_h=all_h,cur_x=all_x,cur_y=all_y)
    #split 120495
    torch.manual_seed(1001)
    train_dataset,test_dataset = random_split(pro_data,[80000,len(pro_data)-80000])
    train_len = 80000
    train_batch_size = 96
    valid_dataset,test_dataset = random_split(test_dataset,[10000,len(test_dataset)-10000])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    test_batch_size = train_batch_size
    valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size, shuffle=False,
                             drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                             drop_last=True)
    my_model = WMEGNN(in_dim=4,out_dim=256,coord_dim=3)
    if torch.cuda.is_available():
        my_model = my_model.cuda()
    # 数据初始化 与  训练优化器 损失函数
    optimizer = torch.optim.AdamW(my_model.parameters(), lr=1e-4)
    # 退火和热重启
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40,T_mult=2, eta_min=1e-9, last_epoch=-1,verbose=False)
    # 单周期
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,
                                                    pct_start=0.20,
                                                    div_factor=1e4,
                                                    anneal_strategy='cos',
                                                    total_steps=600*(train_len//train_batch_size))
    train_loop(
        n_epochs=600,
        train_loader=train_loader,
        valid_loader=valid_loader,
        batch_size=train_batch_size,
        optimizer=optimizer,
        model=my_model,
        loss_f=nn.L1Loss(),
        scheduler=scheduler
    )
    # my_model = torch.load("C:\\Users\\liuyu\\Desktop\\HGPool\\HP_EGNN.pth")
    # torch.save(my_model, "./miu_qm9.pth")
    #test dataset
    with torch.no_grad():
        child = 0
        all_sum_mae = 0
        for t_h_train, t_x_train, t_c_train  in tqdm(test_loader, desc="test dataset looping"):
            t_h_train = t_h_train.float()
            t_x_train = t_x_train.float()
            if torch.cuda.is_available():
                t_h_train = t_h_train.cuda()
                t_x_train = t_x_train.cuda()
                t_c_train = t_c_train.cuda()
            pred = my_model(t_h_train, t_x_train)
            child = child + 1
            mae = torch.abs(pred - t_c_train).sum(dim=-1).sum(dim=0)
            mae = (mae).cpu().detach().numpy()
            all_sum_mae = all_sum_mae + float(float(mae) / float(train_batch_size))
        print(f"cur_test_res: {(all_sum_mae / child):.9f}", all_sum_mae, child)
    #         all_sum_mae = all_sum_mae + float(float(cur_mae) / float(train_batch_size))
    # print(f"final train result: {(all_sum_mae / child):.5f}", all_sum_mae, child)

if __name__ == '__main__':
    main()
    print(torch.__version__)
    print("Task accomplished!!")
