import torch
import torch.nn as nn
class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))



class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types
        # self.BN1 = torch.nn.BatchNorm1d(44,56)
        # self.BN2 = torch.nn.BatchNorm1d(44,56)
        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Tanh()
        )
    #     self._initialization()
    #
    # def _initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             m.weight.data.normal_(0.0, 0.02)
    #             m.bias.data.fill_(0)

    def forward(self, state_in, state_cur, A):
        #A_in = A#[:, :, :self.n_node*self.n_edge_types]

        a_in = torch.bmm(A, state_in)
        a = torch.cat((a_in, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, r * state_cur), 2)
        #joined_input = self.BN2(joined_input)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(GGNN, self).__init__()

        assert (opt.state_dim >= opt.annotation_dim,  \
                'state_dim must be no less than annotation_dim')

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.n_edge_types = opt.edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)

        self.in_fcs = AttrProxy(self, "in_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim * 4, self.state_dim * 4),
            #torch.nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.state_dim * 4, self.state_dim),
            #torch.nn.Dropout(0.1),
            torch.nn.BatchNorm1d(44),
            nn.Tanh(),
            nn.Linear(self.state_dim, self.state_dim),
            nn.ReLU(),
            nn.Linear(self.state_dim, 1),
            nn.ReLU()
        )
        # self.add_node = nn.Sequential(
        #     nn.Linear(self.state_dim, self.state_dim),
        #     nn.Tanh()
      #
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A, y_mask):
        #anno_ones = torch.reshape(annotation, (-1, self.n_node))
        #A = A + torch.eye(self.n_node).repeat(self.n_edge_types,1).T.view(-1,self.n_node*self.n_edge_types).cuda()
        propgate_list = []
        for i_step in range(self.n_steps):
            in_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            # node_state = self.add_node(prop_state)
            prop_state = self.propogator(in_states, prop_state, A)
            propgate_list.append(prop_state)
            # prop_state = prop_state + node_state
        join_state = torch.cat(propgate_list, 2)

        output = self.out(join_state)
        output = torch.reshape(output,(-1,44))
        #output = output.sum(2)
        output = torch.mul(output, y_mask)

        return output


class Propogator1(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator1, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*2+1, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*2+1, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_cur, A, annotation):
        #A_in = A#[:, :, :self.n_node*self.n_edge_types]

        a_in = torch.bmm(A, state_in)
        a = torch.cat((a_in, state_cur), 2)
        a = torch.cat((a,annotation),2)
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output