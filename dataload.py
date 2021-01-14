import pickle as pkl
import numpy as np
import sparse

"""load in data and split into train set(0.8)/ test set"""
def dataloader(file = "d:/rotation2/datasets/processedData13C_NEW.pickle", set_type="", Seed=784):
    with open(file,'rb') as f:
        [DV, DE, DY, DM, _] = pkl.load(f)

    #Seed = np.random.randint(1,1000)
    train_n = 14552
    DV = DV.todense()
    DE = DE.todense()
    DM = DM.todense()
    np.random.seed(Seed)
    np.random.shuffle(DV)
    DV_trn = DV[:train_n, :, :]
    DV_tst = DV[train_n:, :, :]

    np.random.seed(Seed)
    np.random.shuffle(DE)
    DE_trn = DE[:train_n, :, :]
    DE_tst = DE[train_n:, :, :]

    np.random.seed(Seed)
    np.random.shuffle(DY)
    DY_trn = DY[:train_n,:]
    DY_tst = DY[train_n:,:]

    np.random.seed(Seed)
    np.random.shuffle(DM)
    DM_trn = DM[:train_n,:]
    DM_tst = DM[train_n:,:]

    if set_type == 'train':
        return [DV_trn, DE_trn, DY_trn, DM_trn]

    elif set_type == 'val':
        return [DV_tst[:3000], DE_tst[:3000], DY_tst[:3000], DM_tst[:3000]]

    elif set_type == "tst":
        return [DV_tst[3000:], DE_tst[3000:], DY_tst[3000:], DM_tst[3000:]]
# def main():
#     set = dataloader("processedData13C_NEW.pickle","train",784)
#     print(len(set[0]))
# if __name__ == "__main__":
#     main()
