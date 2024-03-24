from __future__ import division
from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
from sklearn import metrics

from config import Config
from models import MAFN
from utils import *


def load_data(dataset):
    print("load data:")
    path = "../generate_data/DLPFC/" + dataset + "/MAFN.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nfadj, nsadj, graph_nei, graph_neg


def train():
    model.train()
    optimizer.zero_grad()
    emb ,pi, disp, mean,emb1,emb2,com= model(features, sadj, fadj)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)

    dcir_loss= dicr_loss(emb1, emb2)


    total_loss =  config.alpha * zinb_loss+config.gamma * reg_loss+0.1*dcir_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss, dcir_loss, total_loss


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # datasets = ['151507', '151508', '151509', '151510', '151669', '151670',
    #             '151671', '151672', '151673', '151674', '151675', '151676']
    datasets = ['151507']
    for i in range(len(datasets)):
        dataset = datasets[i]
        config_file = './config/DLPFC.ini'
        print(dataset)
        adata, features, labels, fadj, sadj, graph_nei, graph_neg = load_data(dataset)
        print(adata)

        plt.rcParams["figure.figsize"] = (3, 3)
        savepath = './result/DLPFC/' + dataset + '/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        title = "Manual annotation (slice #" + dataset + ")"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title,
                      show=False)
        plt.savefig(savepath + 'Manual Annotation.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        config.epochs = 200
        config.epochs = config.epochs + 1

        if cuda:
            features = features.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        import random

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        print(dataset, ' ', config.lr, ' ', config.alpha, ' ', config.beta, ' ', config.gamma)
        model = MAFN(nfeat=config.fdim,
                             nhid1=config.nhid1,
                             nhid2=config.nhid2,

                             dropout=config.dropout)
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        epoch_max = 0
        ari_max = 0
        idx_max = []
        mean_max = []
        emb_max = []

        for epoch in range(config.epochs):
            emb, mean, zinb_loss, reg_loss, dcir_loss,total_loss = train()
            print(dataset, ' epoch: ', epoch,' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss), ' dcir_loss= {:.2f}'.format(dcir_loss),


                  ' total_loss = {:.2f}'.format(total_loss))
            kmeans = KMeans(n_clusters=config.class_num).fit(emb)
            idx = kmeans.labels_
            ari_res = metrics.adjusted_rand_score(labels, idx)
            if ari_res > ari_max:
                ari_max = ari_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb

        print(dataset, ' ', ari_max)

        title = 'MAFN: ARI={:.2f}'.format(ari_max)
        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max

        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        plt.savefig(savepath + 'MAFN.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        sc.pp.neighbors(adata, use_rep='mean')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.tl.paga(adata, groups='idx')
        sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2,
                           show=False)
        plt.savefig(savepath + 'MAFN_umap_mean.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        pd.DataFrame(emb_max).to_csv(savepath + 'MAFN_emb.csv')
        pd.DataFrame(idx_max).to_csv(savepath + 'MAFN_idx.csv')
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        adata.write(savepath + 'MAFN.h5ad')


        sc.pl.umap(adata, color=['idx'], frameon = False)