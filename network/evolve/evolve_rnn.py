import torch.nn as nn
from .utils import prepare_training, prepare_testing_init, img_poly_to_can_poly, get_gcn_feature, get_gcn_feature_window
from .convlstm import ConvLSTM, ConvBLSTM
from .snake import AggregateCirConv
import torch

CUDA_LAUNCH_BLOCKING=1
TYPE_DICT = {'dynamic_hidden_dim': 1,
             'static_hidden_dim': 2}
LSTM_DICT = {'linear': 1,
             'conv': 2}
CONV_FC_DICT = {'pooling_first': 1,
                'dim_reduction_fc': 2,
                'just_fc': 3,
                'fuse_fc': 4}
FC_ACT_DICT = {'none': 1,
               'relu': 2,
               'drop': 3,
               'relu_drop':4}

class ConvLSTMblock(nn.Module):
    def __init__(self, hidden_dims=[128,32], type='dynamic_hidden_dim', num_layers=2, dim_in=64, bidirectional=False,
                 fc_dims=[2], input_size=(7, 7), fc_type='dim_reduction_first', fc_activation='none', drop_factor=0.5):
        super(ConvLSTMblock, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.type_num = TYPE_DICT[type]
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.fc_dims = fc_dims
        self.fc_type = CONV_FC_DICT[fc_type]
        self.fc_activation_type = FC_ACT_DICT[fc_activation]
        # todo : add bidirectional option
        if self.bidirectional:
            setattr(self, f'lstmlayer', ConvBLSTM(input_size=input_size,
                                                 input_dim=dim_in,
                                                 hidden_dim=2*hidden_dims,
                                                 kernel_size=(3, 3),
                                                 num_layers=num_layers,
                                                 batch_first=True,
                                                 bias=True))
        else:
            setattr(self, f'lstmlayer', ConvLSTM(input_size=input_size,
                                                     input_dim=dim_in,
                                                     hidden_dim=hidden_dims,
                                                     kernel_size=(3, 3),
                                                     num_layers=num_layers,
                                                     batch_first=True,
                                                     bias=True))
        # define fc part
        if self.fc_type == 1:
            # type1: global_pooling - fc
            fc_dim_pre = hidden_dims[-1] if isinstance(hidden_dims, (list, tuple)) else hidden_dims
            if self.bidirectional:
                fc_dim_pre = 2 * fc_dim_pre
            for fc_i in range(len(fc_dims)):
                if self.fc_activation_type == 1 or fc_i == len(fc_dims)-1:
                    layer = nn.Linear(fc_dim_pre, fc_dims[fc_i])
                elif self.fc_activation_type == 2:
                    layer = nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                                         nn.ReLU(True))
                elif self.fc_activation_type == 3:
                    layer = nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                          nn.Dropout(drop_factor))
                elif self.fc_activation_type == 4:
                    layer = nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                          nn.ReLU(True),
                                          nn.Dropout(drop_factor))

                setattr(self, f'fc{fc_i}', layer)
                fc_dim_pre = fc_dims[fc_i]
        elif self.fc_type == 2:
            # type2: dim_reduction - flatten - fc
            dim_pre = hidden_dims[-1] if isinstance(hidden_dims, (list, tuple)) else hidden_dims
            if self.bidirectional:
                dim_pre = 2 * dim_pre
            self.dim_reduce = nn.Conv3d(dim_pre, 1,  1)
            fc_dim_pre = input_size[0] * input_size[1]
            for fc_i in range(len(fc_dims)):
                if self.fc_activation_type == 1 or fc_i == len(fc_dims)-1:
                    layer = nn.Linear(fc_dim_pre, fc_dims[fc_i])
                elif self.fc_activation_type == 2:
                    layer = nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                          nn.ReLU(True))
                elif self.fc_activation_type == 3:
                    layer = nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                          nn.Dropout(drop_factor))
                elif self.fc_activation_type == 4:
                    layer = nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                          nn.ReLU(True),
                                          nn.Dropout(drop_factor))

                setattr(self, f'fc{fc_i}', layer)
                fc_dim_pre = fc_dims[fc_i]
        elif self.fc_type == 3:
            # type3: flatten - fc
            fc_dim_pre = input_size[0] * input_size[1] * (hidden_dims[-1] if isinstance(hidden_dims, (list, tuple)) else hidden_dims)
            if self.bidirectional:
                fc_dim_pre = 2 * fc_dim_pre
            # self.dim_reduce = nn.Conv3d(dim_pre, 1, 1)
            for fc_i in range(len(fc_dims)):
                if self.fc_activation_type == 1 or fc_i == len(fc_dims)-1:
                    layer = nn.Linear(fc_dim_pre, fc_dims[fc_i])
                elif self.fc_activation_type == 2:
                    layer = nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                          nn.ReLU(True))
                elif self.fc_activation_type == 3:
                    layer = nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                          nn.Dropout(drop_factor))
                elif self.fc_activation_type == 4:
                    layer = nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                          nn.ReLU(True),
                                          nn.Dropout(drop_factor))
                setattr(self, f'fc{fc_i}', layer)
                fc_dim_pre = fc_dims[fc_i]
        elif self.fc_type == 4:
            # type4: fuse - flatten - fc
            fc_dim_pre = input_size[0] * input_size[1] * (
                hidden_dims[-1] if isinstance(hidden_dims, (list, tuple)) else hidden_dims)
            if self.bidirectional:
                self.fuse = nn.Conv2d(2 * (hidden_dims[-1] if isinstance(hidden_dims, (list, tuple)) else hidden_dims), hidden_dims[-1] if isinstance(hidden_dims, (list, tuple)) else hidden_dims, kernel_size=1, padding=0, bias=True)
            # self.dim_reduce = nn.Conv3d(dim_pre, 1, 1)
            for fc_i in range(len(fc_dims)):
                if self.fc_activation_type == 1 or fc_i == len(fc_dims)-1:
                    layer = nn.Linear(fc_dim_pre, fc_dims[fc_i])
                elif self.fc_activation_type == 2:
                    layer = nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                          nn.ReLU(True))
                elif self.fc_activation_type == 3:
                    layer = nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                          nn.Dropout(drop_factor))
                elif self.fc_activation_type == 4:
                    layer = nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                          nn.ReLU(True),
                                          nn.Dropout(drop_factor))
                setattr(self, f'fc{fc_i}', layer)
                fc_dim_pre = fc_dims[fc_i]


    def forward(self, x):
        out_lstm = x #NxLxCxHxW
        h_f = torch.zeros(self.num_layers, x.size(0), self.hidden_dims[-1] if isinstance(self.hidden_dims, (list, tuple)) else self.hidden_dims, x.size(3), x.size(4)).to(x.device)
        c_f = torch.zeros(self.num_layers, x.size(0), self.hidden_dims[-1] if isinstance(self.hidden_dims, (list, tuple)) else self.hidden_dims, x.size(3), x.size(4)).to(x.device)
        if self.bidirectional:
            h_b = torch.zeros(self.num_layers, x.size(0),
                              self.hidden_dims[-1] if isinstance(self.hidden_dims, (list, tuple)) else self.hidden_dims,
                              x.size(3), x.size(4)).to(x.device)
            c_b = torch.zeros(self.num_layers, x.size(0),
                              self.hidden_dims[-1] if isinstance(self.hidden_dims, (list, tuple)) else self.hidden_dims,
                              x.size(3), x.size(4)).to(x.device)
        if self.bidirectional:
            out_lstm = getattr(self, 'lstmlayer')(out_lstm, torch.flip(out_lstm,[1]), (h_f, c_f), (h_b, c_b))  # out_lstm : N x L x c x h x w
        else:
            out_lstm, last_state = getattr(self, 'lstmlayer')(out_lstm, (h_f, c_f)) #out_lstm : N x L x c x h x w
        if isinstance(out_lstm, list):
            out_lstm = out_lstm[-1]

        if self.fc_type == 1:
            out_avg_pooled = out_lstm.mean(-1).mean(-1) #out_avg_pooled : N x L x c
            pre_hidden = out_avg_pooled
            for fc_i in range(len(self.fc_dims)):
                out_fc = self.__getattr__(f'fc{fc_i}')(pre_hidden) #out_avg_pooled : NxLxc
                pre_hidden = out_fc
            out = pre_hidden
        elif self.fc_type == 2:
            out_dim_reduce = self.dim_reduce(out_lstm.permute(0, 2, 1, 3, 4)) # N x 1 x L x h x w
            out_flatten = torch.flatten(out_dim_reduce.squeeze(1), start_dim=2) # N x L x h x w
            pre_hidden = out_flatten
            for fc_i in range(len(self.fc_dims)):
                out_fc = self.__getattr__(f'fc{fc_i}')(pre_hidden) #out_avg_pooled : NxLxc
                pre_hidden = out_fc
            out = pre_hidden
        elif self.fc_type == 3:
            out_flatten = torch.flatten(out_lstm, start_dim=2)  # N x L x c x h x w
            pre_hidden = out_flatten
            for fc_i in range(len(self.fc_dims)):
                out_fc = self.__getattr__(f'fc{fc_i}')(pre_hidden)  # out_avg_pooled : NxLxc
                pre_hidden = out_fc
            out = pre_hidden
        elif self.fc_type == 4:
            if self.bidirectional:
                out_lstm_reshape = out_lstm.reshape(out_lstm.size(0)*out_lstm.size(1), out_lstm.size(2), out_lstm.size(3), out_lstm.size(4))# N x L x 2c x h x w -> N x L x c x h x w
                out_lstm_fuse = self.fuse(out_lstm_reshape)
            else:
                out_lstm_fuse = out_lstm
            out_flatten = torch.flatten(out_lstm_fuse.reshape(out_lstm.size(0), out_lstm.size(1), out_lstm_fuse.size(1), out_lstm.size(3), out_lstm.size(4)), start_dim=2)  # N x L x c x h x w
            pre_hidden = out_flatten
            for fc_i in range(len(self.fc_dims)):
                out_fc = self.__getattr__(f'fc{fc_i}')(pre_hidden)  # out_avg_pooled : NxLxc
                pre_hidden = out_fc
            out = pre_hidden
        return out

class LSTMblock(nn.Module):
    def __init__(self, hidden_dims=[128,32], type='dynamic_hidden_dim', num_layers=2, dim_in=64, bidirectional=False,
                 fc_dims=[2]):
        super(LSTMblock, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.type_num = TYPE_DICT[type]
        self.bidirectional = bidirectional
        self.fc_dims = fc_dims
        if self.bidirectional:
            self.D = 2
        else:
            self.D = 1
        dim_pre = dim_in

        if self.type_num == 1:
            for id, dim in enumerate(hidden_dims):
                setattr(self, f'lstmlayer{id}', nn.LSTM(dim_pre, dim, batch_first=True, bidirectional=bidirectional))
                dim_pre = dim
        else:
            setattr(self, f'lstmlayer',
                    nn.LSTM(dim_pre, hidden_dims[-1] if isinstance(hidden_dims, (list, tuple)) else hidden_dims,
                            batch_first=True, num_layers=self.num_layers, bidirectional=bidirectional))
        fc_dim_pre = self.D * (hidden_dims[-1] if isinstance(hidden_dims, (list, tuple)) else hidden_dims)

        for fc_i in range(len(fc_dims)):
            setattr(self, f'fc{fc_i}', nn.Linear(fc_dim_pre, fc_dims[fc_i]))
            fc_dim_pre = fc_dims[fc_i]

    def forward(self, x):
        out_lstm = x.permute(0,2,1)
        if self.type_num == 1:
            for id in range(len(self.hidden_dims)):
                h = torch.zeros(1*self.D, x.size(0), self.hidden_dims[id]).to(x.device)
                c = torch.zeros(1*self.D, x.size(0), self.hidden_dims[id]).to(x.device)
                out_lstm, (h, c) = getattr(self, f'lstmlayer{id}')(out_lstm, (h, c))
        else:
            h = torch.zeros(self.num_layers*self.D, x.size(0), self.hidden_dims[-1] if isinstance(self.hidden_dims, (list, tuple)) else self.hidden_dims).to(x.device)
            c = torch.zeros(self.num_layers*self.D, x.size(0), self.hidden_dims[-1] if isinstance(self.hidden_dims, (list, tuple)) else self.hidden_dims).to(x.device)
            if x.size(0) > 0:
                out_lstm, (h, c) = getattr(self, 'lstmlayer')(out_lstm, (h, c))
            else:
                out_lstm = out_lstm.reshape((out_lstm.size(0), out_lstm.size(1), self.D * h.size(-1)))

        for fc_i in range(len(self.fc_dims)):
            out_lstm = getattr(self, f'fc{fc_i}')(out_lstm)
        return out_lstm

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_dims=[64,16], type='dynamic_hidden_dim', num_layers=2, num_dec=1, ro=4.,
                 use_input_with_rel_ind=False, is_dec_exclusive=True, bidirectional=False, fc_dims=[2],
                 lstm_type='linear', lstm_input_window_size=(7, 7), lstm_input_window_stride=1, lstm_fc_type='dim_reduction_first',
                 lstm_fc_activation='none', grad_feature_neighbors=True, c_in=64, num_points=128):
        super(DecoderLSTM, self).__init__()
        self.num_points = num_points
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.num_dec = num_dec
        self.type_num = TYPE_DICT[type]
        self.ro = ro
        self.use_input_with_rel_ind = use_input_with_rel_ind
        self.is_dec_exclusive = is_dec_exclusive
        self.lstm_type = LSTM_DICT[lstm_type]
        self.lstm_input_window_size = lstm_input_window_size
        self.lstm_input_window_stride = lstm_input_window_stride
        self.grad_feature_neighbors = grad_feature_neighbors

        if self.use_input_with_rel_ind:
            dim_in = c_in+2
        else:
            dim_in = c_in

        if self.lstm_type == 1:
            if self.is_dec_exclusive:
                for dec in range(num_dec):
                    setattr(self, f'dec{dec}', LSTMblock(hidden_dims=hidden_dims, type=type, num_layers=num_layers, dim_in=dim_in,
                                                         bidirectional=bidirectional, fc_dims=fc_dims))
            else:
                setattr(self, f'dec', LSTMblock(hidden_dims=hidden_dims, type=type, num_layers=num_layers, dim_in=dim_in,
                                                         bidirectional=bidirectional, fc_dims=fc_dims))
        else:
            if self.is_dec_exclusive:
                for dec in range(num_dec):
                    setattr(self, f'dec{dec}', ConvLSTMblock(hidden_dims=hidden_dims, type=type, num_layers=num_layers, dim_in=dim_in,
                                                         bidirectional=bidirectional, fc_dims=fc_dims, input_size=lstm_input_window_size,
                                                         fc_type=lstm_fc_type, fc_activation=lstm_fc_activation))
            else:
                setattr(self, f'dec',
                        ConvLSTMblock(hidden_dims=hidden_dims, type=type, num_layers=num_layers, dim_in=dim_in,
                                      bidirectional=bidirectional, fc_dims=fc_dims, input_size=lstm_input_window_size,
                                      fc_type=lstm_fc_type, fc_activation=lstm_fc_activation))

    def prepare_training(self, output, batch):
        init = prepare_training(output, batch, self.ro, num_points=self.num_points)
        return init

    def prepare_testing_init(self, output):
        init = prepare_testing_init(output['poly_coarse'], self.ro, num_points=self.num_points)
        return init

    def prepare_testing_evolve(self, output, h, w):
        img_init_polys = output['img_init_polys']
        img_init_polys[..., 0] = torch.clamp(img_init_polys[..., 0], min=0, max=w-1)
        img_init_polys[..., 1] = torch.clamp(img_init_polys[..., 1], min=0, max=h-1)
        output.update({'img_init_polys': img_init_polys})
        return img_init_polys

    def forward_train(self, output, batch, cnn_feature):
        ret = output
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        #prepare
        init = self.prepare_training(output, batch)
        #decoder - LSTM
        py_pred = init['img_init_polys']
        py_preds = []
        for dec in range(self.num_dec):
            if self.lstm_type == 1:
                out_lstm = get_gcn_feature(cnn_feature, py_pred, init['py_ind'], h, w)
                if self.use_input_with_rel_ind:
                    c_py_pred = img_poly_to_can_poly(py_pred)
                    out_lstm = torch.cat([out_lstm, c_py_pred.permute(0, 2, 1)], dim=1)
            else:
                out_lstm = get_gcn_feature_window(cnn_feature, py_pred, init['py_ind'], h, w,
                                                  self.lstm_input_window_stride, self.lstm_input_window_size,
                                                  grad_neighbors=self.grad_feature_neighbors)
                out_lstm = out_lstm.permute(0, 2, 1, 3, 4) #NxLxCxHxW
                if self.use_input_with_rel_ind:
                    c_py_pred = img_poly_to_can_poly(py_pred).unsqueeze(-1).unsqueeze(-1) #NxLxCxHxW
                    out_lstm = torch.cat([out_lstm, c_py_pred.expand(-1, -1, -1, out_lstm.size(3), out_lstm.size(4))], dim=2)

            if self.is_dec_exclusive:
                offset = getattr(self, f'dec{dec}')(out_lstm)
            else:
                offset = getattr(self, f'dec')(out_lstm)

            py_pred = init['img_init_polys'].detach() + offset
            py_preds.append(py_pred*self.ro)
        ret.update({'py_pred': py_preds, 'batch_ind': init['py_ind'].to(device=out_lstm.device)})
        ret.update({'img_gt_polys': init['img_gt_polys'] * self.ro})
        ret.update({'img_gt_init_polys': init['img_gt_init_polys'] * self.ro})
        ret.update({'img_gt_coarse_polys': init['img_gt_coarse_polys'] * self.ro})
        return output

    def forward_test(self, output, cnn_feature):
        ret = output
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        with torch.no_grad():
            #prepare
            init = self.prepare_testing_init(output)
            img_init_polys = self.prepare_testing_evolve(init, cnn_feature.size(2), cnn_feature.size(3))
            # decoder - LSTM
            py = img_init_polys
            pys = []
            for dec in range(self.num_dec):
                if self.lstm_type == 1:
                    out_lstm = get_gcn_feature(cnn_feature, py, init['py_ind'], h, w)
                    if self.use_input_with_rel_ind:
                        c_py = img_poly_to_can_poly(py)
                        out_lstm = torch.cat([out_lstm, c_py.permute(0, 2, 1)], dim=1)
                else:
                    out_lstm = get_gcn_feature_window(cnn_feature, py, init['py_ind'], h, w,
                                                      self.lstm_input_window_stride, self.lstm_input_window_size,
                                                      grad_neighbors=self.grad_feature_neighbors)
                    out_lstm = out_lstm.permute(0, 2, 1, 3, 4)  # NxLxCxHxW
                    if self.use_input_with_rel_ind:
                        c_py = img_poly_to_can_poly(py).unsqueeze(-1).unsqueeze(-1)
                        out_lstm = torch.cat([out_lstm, c_py.expand(-1, -1, -1, out_lstm.size(3), out_lstm.size(4))], dim=2)
                if self.is_dec_exclusive:
                    offset = getattr(self, f'dec{dec}')(out_lstm)
                else:
                    offset = getattr(self, f'dec')(out_lstm)
                py = img_init_polys.detach() + offset
                pys.append(py*self.ro)
            ret.update({'py': pys, 'batch_ind': init['py_ind'].to(device=out_lstm.device)})
        return output

    def forward(self, output, cnn_feature, batch=None):
        if batch is not None and 'test' not in batch['meta']:
            self.forward_train(output, batch, cnn_feature)
        else:
            self.forward_test(output, cnn_feature)
        return output


class RNNblock(nn.Module):
    def __init__(self, hidden_dims=[128,32], type='dynamic_hidden_dim', num_layers=2, dim_in=64, bidirectional=False,
                 fc_dims=[2], rnn_type='GRU', fc_activation='none', drop_factor=0.5, bi_comb_type='none'):
        super(RNNblock, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.type_num = TYPE_DICT[type]
        self.bidirectional = bidirectional
        self.fc_dims = fc_dims
        self.rnn_type = rnn_type
        if self.bidirectional:
            self.D = 2
        else:
            self.D = 1
        self.fc_activation_type = FC_ACT_DICT[fc_activation]
        self.bi_comb_type = bi_comb_type
        dim_pre = dim_in

        if self.type_num == 1:
            for id, dim in enumerate(hidden_dims):
                dim_out = dim
                setattr(self, f'rnnlayer{id}', getattr(nn,rnn_type)(input_size=dim_pre, hidden_size=dim_out, batch_first=True, bidirectional=bidirectional))
                if self.bidirectional and (self.bi_comb_type == 'conv'):
                    setattr(self, f'rnncomb{id}', nn.Conv1d(dim_out*2, dim_out, 1))
                dim_pre = dim_out*self.D if self.bi_comb_type=='none' else dim_out
            if self.bidirectional and (self.bi_comb_type == 'none'):
                setattr(self, f'rnncomb{len(hidden_dims)-1}', nn.Conv1d(hidden_dims[-1] * 2, hidden_dims[-1], 1))
        else:
            setattr(self, f'rnnlayer',
                    getattr(nn,rnn_type)(dim_pre, hidden_dims[-1] if isinstance(hidden_dims, (list, tuple)) else hidden_dims,
                            batch_first=True, num_layers=self.num_layers, bidirectional=bidirectional))
        fc_dim_pre = self.D * (hidden_dims[-1] if isinstance(hidden_dims, (list, tuple)) else hidden_dims)

        for fc_i in range(len(fc_dims)):
            if self.fc_activation_type == 1 or fc_i == len(fc_dims) - 1:
                setattr(self, f'fc{fc_i}', nn.Linear(fc_dim_pre, fc_dims[fc_i]))
            elif self.fc_activation_type == 2:
                setattr(self, f'fc{fc_i}', nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                      nn.ReLU(True)))
            elif self.fc_activation_type == 3:
                setattr(self, f'fc{fc_i}', nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                      nn.Dropout(drop_factor)))
            elif self.fc_activation_type == 4:
                setattr(self, f'fc{fc_i}', nn.Sequential(nn.Linear(fc_dim_pre, fc_dims[fc_i]),
                                      nn.ReLU(True),
                                      nn.Dropout(drop_factor)))
            # setattr(self, f'fc{fc_i}', nn.Linear(fc_dim_pre, fc_dims[fc_i]))
            fc_dim_pre = fc_dims[fc_i]

    def forward(self, x):
        out_rnn = x.permute(0,2,1)
        if x.size(0) > 0:
            if self.type_num == 1:
                for id in range(len(self.hidden_dims)):
                    h = torch.zeros(1 * self.D, x.size(0), self.hidden_dims[id]).to(x.device)
                    if self.rnn_type == 'LSTM':
                        c = torch.zeros(1*self.D, x.size(0), self.hidden_dims[id]).to(x.device)
                        out_rnn, (h, c) = getattr(self, f'rnnlayer{id}')(out_rnn, (h, c))
                    else:
                        out_rnn, h_o = getattr(self, f'rnnlayer{id}')(out_rnn, h)
                    if self.bidirectional and (self.bi_comb_type == 'conv'):
                        out_rnn = out_rnn.permute(0, 2, 1)
                        out_rnn = getattr(self, f'rnncomb{id}')(out_rnn)
                        out_rnn = out_rnn.permute(0, 2, 1)
                    # torch.cuda.synchronize()
                if self.bidirectional and (self.bi_comb_type == 'none'):
                    out_rnn = out_rnn.permute(0, 2, 1)
                    out_rnn = getattr(self, f'rnncomb{len(self.hidden_dims)-1}')(out_rnn)
                    out_rnn = out_rnn.permute(0, 2, 1)
            else:
                h = torch.zeros(self.num_layers*self.D, x.size(0), self.hidden_dims[-1] if isinstance(self.hidden_dims, (list, tuple)) else self.hidden_dims).to(x.device)
                c = torch.zeros(self.num_layers*self.D, x.size(0), self.hidden_dims[-1] if isinstance(self.hidden_dims, (list, tuple)) else self.hidden_dims).to(x.device)
                if x.size(0) > 0:
                    if self.rnn_type == 'LSTM':
                        out_rnn, (h, c) = getattr(self, 'rnnlayer')(out_rnn, (h, c))
                    else:
                        out_rnn, h = getattr(self, 'rnnlayer')(out_rnn, h)
                else:
                    out_rnn = out_rnn.reshape((out_rnn.size(0), out_rnn.size(1), self.D * h.size(-1)))

            for fc_i in range(len(self.fc_dims)):
                out_rnn = getattr(self, f'fc{fc_i}')(out_rnn)
        else:
            out_rnn = torch.zeros(out_rnn.size(0),out_rnn.size(1),2).to(x.device)
        return out_rnn

class RefineRNN(nn.Module):
    def __init__(self, hidden_dims=[64,16], type='dynamic_hidden_dim', num_layers=2, num_dec=1, ro=4.,
                 use_input_with_rel_ind=False, is_dec_exclusive=True, bidirectional=False, fc_dims=[2],
                 layer_type='linear', input_window_size=(7, 7), input_window_stride=1, fc_type='dim_reduction_first',
                 fc_activation='none', grad_feature_neighbors=True, c_in=64, num_points=128, rnn_type='GRU',
                 aggregate_feature=False, refine_kernel_size=3, aggregate_type='default',
                 aggregate_fusion_conv_num=3, aggregate_fusion_state_dim=256, bi_comb_type='none'):
        super(RefineRNN, self).__init__()
        self.num_points = num_points
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.num_dec = num_dec
        self.type_num = TYPE_DICT[type]
        self.ro = ro
        self.use_input_with_rel_ind = use_input_with_rel_ind
        self.is_dec_exclusive = is_dec_exclusive
        self.layer_type = LSTM_DICT[layer_type]
        self.input_window_size = input_window_size
        self.input_window_stride = input_window_stride
        self.grad_feature_neighbors = grad_feature_neighbors
        self.rnn_type = rnn_type
        self.aggregate_feature = aggregate_feature
        self.aggregate_fusion_conv_num = aggregate_fusion_conv_num
        self.aggregate_fusion_state_dim = aggregate_fusion_state_dim

        if self.use_input_with_rel_ind:
            dim_in = c_in+2
        else:
            dim_in = c_in

        if self.layer_type == 1:
            if self.is_dec_exclusive:
                for dec in range(num_dec):
                    if self.aggregate_feature:
                        setattr(self, f'aggregate{dec}', AggregateCirConv(state_dim=128, feature_dim=64, conv_type='dgrid',
                                                          refine_kernel_size=refine_kernel_size, type=aggregate_type,
                                                                          fusion_conv_num=aggregate_fusion_conv_num, fusion_state_dim=aggregate_fusion_state_dim))

                    setattr(self, f'dec{dec}', RNNblock(hidden_dims=hidden_dims, type=type, num_layers=num_layers, dim_in=dim_in,
                                                         bidirectional=bidirectional, fc_dims=fc_dims, rnn_type=self.rnn_type, fc_activation=fc_activation, bi_comb_type=bi_comb_type))
            else:
                if self.aggregate_feature:
                    setattr(self, f'aggregate', AggregateCirConv(state_dim=128, feature_dim=64, conv_type='dgrid',
                                                                      refine_kernel_size=refine_kernel_size, type=aggregate_type,
                                                                 fusion_conv_num=aggregate_fusion_conv_num, fusion_state_dim=aggregate_fusion_state_dim))
                setattr(self, f'dec', RNNblock(hidden_dims=hidden_dims, type=type, num_layers=num_layers, dim_in=dim_in,
                                                         bidirectional=bidirectional, fc_dims=fc_dims, rnn_type=self.rnn_type, fc_activation=fc_activation, bi_comb_type=bi_comb_type))
        else:
            if self.is_dec_exclusive:
                for dec in range(num_dec):
                    if self.aggregate_feature:
                        setattr(self, f'aggregate{dec}', AggregateCirConv(state_dim=128, feature_dim=64, conv_type='dgrid',
                                                          refine_kernel_size=refine_kernel_size, type=aggregate_type,
                                                                          fusion_conv_num=aggregate_fusion_conv_num, fusion_state_dim=aggregate_fusion_state_dim))
                    setattr(self, f'dec{dec}', ConvLSTMblock(hidden_dims=hidden_dims, type=type, num_layers=num_layers, dim_in=dim_in,
                                                         bidirectional=bidirectional, fc_dims=fc_dims, input_size=input_window_size,
                                                         fc_type=fc_type, fc_activation=fc_activation))
            else:
                if self.aggregate_feature:
                    setattr(self, f'aggregate', AggregateCirConv(state_dim=128, feature_dim=64, conv_type='dgrid',
                                                                      refine_kernel_size=refine_kernel_size, type=aggregate_type,
                                                                 fusion_conv_num=aggregate_fusion_conv_num, fusion_state_dim=aggregate_fusion_state_dim))
                setattr(self, f'dec',
                        ConvLSTMblock(hidden_dims=hidden_dims, type=type, num_layers=num_layers, dim_in=dim_in,
                                      bidirectional=bidirectional, fc_dims=fc_dims, input_size=input_window_size,
                                      fc_type=fc_type, fc_activation=fc_activation))

    def prepare_training(self, output, batch):
        init = prepare_training(output, batch, self.ro, num_points=self.num_points)
        return init

    def prepare_testing_init(self, output):
        init = prepare_testing_init(output['poly_coarse'], self.ro, num_points=self.num_points)
        return init

    def prepare_testing_evolve(self, output, h, w):
        img_init_polys = output['img_init_polys']
        img_init_polys[..., 0] = torch.clamp(img_init_polys[..., 0], min=0, max=w-1)
        img_init_polys[..., 1] = torch.clamp(img_init_polys[..., 1], min=0, max=h-1)
        output.update({'img_init_polys': img_init_polys})
        return img_init_polys

    def forward_train(self, output, batch, cnn_feature):
        ret = output
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        #prepare
        init = self.prepare_training(output, batch)
        #decoder - LSTM
        py_pred = init['img_init_polys']
        py_preds = []
        for dec in range(self.num_dec):
            if self.layer_type == 1:
                out_lstm = get_gcn_feature(cnn_feature, py_pred, init['py_ind'], h, w)
                if self.aggregate_feature:
                    if self.is_dec_exclusive:
                        out_lstm = getattr(self, f'aggregate{dec}')(out_lstm)
                    else:
                        out_lstm = getattr(self, f'aggregate')(out_lstm)
                if self.use_input_with_rel_ind:
                    c_py_pred = img_poly_to_can_poly(py_pred)
                    out_lstm = torch.cat([out_lstm, c_py_pred.permute(0, 2, 1)], dim=1)
            else:
                out_lstm = get_gcn_feature_window(cnn_feature, py_pred, init['py_ind'], h, w,
                                                  self.input_window_stride, self.input_window_size,
                                                  grad_neighbors=self.grad_feature_neighbors)
                if self.aggregate_feature:
                    if self.is_dec_exclusive:
                        out_lstm = getattr(self, f'aggregate{dec}')(out_lstm)
                    else:
                        out_lstm = getattr(self, f'aggregate')(out_lstm)
                out_lstm = out_lstm.permute(0, 2, 1, 3, 4) #NxLxCxHxW
                if self.use_input_with_rel_ind:
                    c_py_pred = img_poly_to_can_poly(py_pred).unsqueeze(-1).unsqueeze(-1) #NxLxCxHxW
                    out_lstm = torch.cat([out_lstm, c_py_pred.expand(-1, -1, -1, out_lstm.size(3), out_lstm.size(4))], dim=2)

            if self.is_dec_exclusive:
                offset = getattr(self, f'dec{dec}')(out_lstm)
            else:
                offset = getattr(self, f'dec')(out_lstm)

            py_pred = init['img_init_polys'].detach() + offset
            py_preds.append(py_pred*self.ro)
        ret.update({'py_pred': py_preds, 'batch_ind': init['py_ind'].to(device=out_lstm.device)})
        ret.update({'img_gt_polys': init['img_gt_polys'] * self.ro})
        ret.update({'img_gt_init_polys': init['img_gt_init_polys'] * self.ro})
        ret.update({'img_gt_coarse_polys': init['img_gt_coarse_polys'] * self.ro})
        return output

    def forward_test(self, output, cnn_feature):
        ret = output
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        with torch.no_grad():
            #prepare
            init = self.prepare_testing_init(output)
            img_init_polys = self.prepare_testing_evolve(init, cnn_feature.size(2), cnn_feature.size(3))
            # decoder - LSTM
            py = img_init_polys
            pys = []
            for dec in range(self.num_dec):
                if self.layer_type == 1:
                    out_lstm = get_gcn_feature(cnn_feature, py, init['py_ind'], h, w)
                    if self.aggregate_feature:
                        if self.is_dec_exclusive:
                            out_lstm = getattr(self, f'aggregate{dec}')(out_lstm)
                        else:
                            out_lstm = getattr(self, f'aggregate')(out_lstm)
                    if self.use_input_with_rel_ind:
                        c_py = img_poly_to_can_poly(py)
                        out_lstm = torch.cat([out_lstm, c_py.permute(0, 2, 1)], dim=1)
                else:
                    out_lstm = get_gcn_feature_window(cnn_feature, py, init['py_ind'], h, w,
                                                      self.input_window_stride, self.input_window_size,
                                                      grad_neighbors=self.grad_feature_neighbors)
                    if self.aggregate_feature:
                        if self.is_dec_exclusive:
                            out_lstm = getattr(self, f'aggregate{dec}')(out_lstm)
                        else:
                            out_lstm = getattr(self, f'aggregate')(out_lstm)
                    out_lstm = out_lstm.permute(0, 2, 1, 3, 4)  # NxLxCxHxW
                    if self.use_input_with_rel_ind:
                        c_py = img_poly_to_can_poly(py).unsqueeze(-1).unsqueeze(-1)
                        out_lstm = torch.cat([out_lstm, c_py.expand(-1, -1, -1, out_lstm.size(3), out_lstm.size(4))], dim=2)
                if self.is_dec_exclusive:
                    offset = getattr(self, f'dec{dec}')(out_lstm)
                else:
                    offset = getattr(self, f'dec')(out_lstm)
                py = img_init_polys.detach() + offset
                pys.append(py*self.ro)
            ret.update({'py': pys, 'batch_ind': init['py_ind'].to(device=out_lstm.device)})
        return output

    def forward(self, output, cnn_feature, batch=None):
        if batch is not None and 'test' not in batch['meta']:
            self.forward_train(output, batch, cnn_feature)
        else:
            self.forward_test(output, cnn_feature)
        return output