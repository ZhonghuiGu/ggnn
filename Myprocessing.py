import numpy as np
import pickle as pkl
import os, sys
import sparse
from Myutil import atomFeatures, bondFeatures
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures

with open('cluster80_13C.pickle', 'rb') as f:
    molset, nmrset = pkl.load(f)


n_max = 64
heavy_max = 44
dim_node = 29
dim_edge = 10
atom_list = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl']

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


DV = []
DE = []
DY = []
DM = []
Dsmi = []
for i, mol in enumerate(molset):
    if '.' in Chem.MolToSmiles(mol):
        continue

    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    Chem.rdmolops.AssignStereochemistry(mol)

    n_atom = mol.GetNumAtoms()

    rings = mol.GetRingInfo().AtomRings()

    feats = chem_feature_factory.GetFeaturesForMol(mol)
    donor_list = []
    acceptor_list = []
    for j in range(len(feats)):
        if feats[j].GetFamily() == 'Donor':
            assert len(feats[j].GetAtomIds()) == 1
            donor_list.append(feats[j].GetAtomIds()[0])
        elif feats[j].GetFamily() == 'Acceptor':
            assert len(feats[j].GetAtomIds()) == 1
            acceptor_list.append(feats[j].GetAtomIds()[0])

    # node DV
    node = np.zeros((n_max, dim_node), dtype=np.int8)
    for j in range(n_atom):
        node[j, :] = atomFeatures(j, mol, rings, atom_list, donor_list, acceptor_list)

    # edge DE
    edge = np.zeros((n_max, n_max, dim_edge), dtype=np.int8)  # j*k*10
    for j in range(n_atom - 1):
        for k in range(j + 1, n_atom):
            edge[j, k, :] = bondFeatures(j, k, mol, rings)
            edge[k, j, :] = edge[j, k, :]

    # property DY and mask DM
    props = nmrset[i]
    mask = np.zeros((n_max, 1), dtype=np.int8)

    property = []  # nmr data
    C_num = 0
    for j in range(n_atom):
        #atom_property = []
        if mol.GetAtomWithIdx(j).GetAtomicNum() == 6:
            C_num += 1
        if j in props:
            property.append(props[j])
            mask[j] = 1
            assert mol.GetAtomWithIdx(j).GetAtomicNum() == 6
        else:
            property.append(0)

    # if C_num != len(props):
    #     print(i,C_num,len(props))
    #     continue
        #property.append(atom_property)
    property = np.array(property)

    # compression
    del_ids = np.where(node[:, 0] == 1)[0]

    node = np.delete(node, del_ids, 0)  # delete where atom is H
    node = np.delete(node, [0], 1)  # since H is empty,remove the H row in atom_list one_hot
    edge = np.delete(edge, del_ids, 0)
    edge = np.delete(edge, del_ids, 1)
    property = np.delete(property, del_ids, 0)
    mask = np.delete(mask, del_ids, 0)

    node = np.pad(node, ((0, n_max - node.shape[0]), (0, 0)))
    edge = np.pad(edge, ((0, n_max - edge.shape[0]), (0, n_max - edge.shape[1]), (0, 0)))
    mask = np.pad(mask, ((0, n_max - mask.shape[0]), (0, 0)))
    property = np.pad(property,(0,(44-property.shape[0])))
    # append
    DV.append(np.array(node))
    DE.append(np.array(edge))
    DY.append(np.array(property))
    DM.append(np.array(mask))
    Dsmi.append(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol))))

    if i % 1000 == 0:
        print(i, flush=True)

# np array
DV = np.asarray(DV, dtype=np.int8)
DE = np.asarray(DE, dtype=np.int8)
DY = np.asarray(DY)
DM = np.asarray(DM, dtype=np.int8)
Dsmi = np.asarray(Dsmi)

DV = DV[:, :heavy_max, :]
DE = DE[:, :heavy_max, :heavy_max, :]
DM = DM[:, :heavy_max, :]


DE_new = np.zeros((20552,44,440))
for i in range(len(DE)):
    DE_=DE[i,:,:,:]
    DE_new[i,:,:] = np.concatenate([DE_[:,:,j] for j in range(10)],1)
print(DE_new[4])
print(DV.shape, DE_new.shape, DY.shape, DM.shape)
# compression
DV = sparse.COO.from_numpy(DV)
DE_new = sparse.COO.from_numpy(DE_new)
DM = sparse.COO.from_numpy(DM)
print(DY[4], Dsmi[4], DM[4])
print(len(molset),len(nmrset),len(DM))


# save
with open('processedData13C_NEW.pickle', 'wb') as fw:
    pkl.dump([DV, DE_new, DY, DM, Dsmi], fw)