import torch
from torch import nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur],
                             dim=1)  # concatenate along channel axis (bxcxhxw)

        combined_conv = self.conv(combined) # c=4 * self.hidden_dim
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim,
                                             dim=1) # c=self.hidden_dim/self.hidden_dim/self.hidden_dim/self.hidden_dim
        i = torch.sigmoid(cc_i) # c=self.hidden_dim
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g # c=self.hidden_dim
        h_next = o * torch.tanh(c_next) # c=self.hidden_dim

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height,
                                     self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height,
                                     self.width)).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size,
                 num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists
        # having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[
                i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output_list (num_layers x seg_len x b x c x h x w)
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            pass
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        h_all, c_all = hidden_state
        for layer_idx in range(self.num_layers):

            h, c = h_all[layer_idx], c_all[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all(
                    [isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvBLSTM(nn.Module):
    # Constructor
    def __init__(self, input_size, input_dim, hidden_dim,
                 kernel_size, num_layers, bias=True, batch_first=False):
        super(ConvBLSTM, self).__init__()
        self.forward_net = ConvLSTM(input_size, input_dim, hidden_dim // 2, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias)
        self.reverse_net = ConvLSTM(input_size, input_dim, hidden_dim // 2, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias)

    def forward(self, xforward, xreverse, hidden_forward=None, hidden_reverse=None):
        """
        xforward, xreverse = B T C H W tensors.
        """

        y_out_fwd, _ = self.forward_net(xforward, hidden_forward)
        y_out_rev, _ = self.reverse_net(xreverse, hidden_reverse)

        y_out_fwd = y_out_fwd[-1]  # outputs of last CLSTM layer = B, T, C, H, W
        y_out_rev = y_out_rev[-1]  # outputs of last CLSTM layer = B, T, C, H, W

        reversed_idx = list(reversed(range(y_out_rev.shape[1])))
        y_out_rev = y_out_rev[:, reversed_idx, ...]  # reverse temporal outputs.
        ycat = torch.cat((y_out_fwd, y_out_rev), dim=2)

        return ycat

class AttentionWeightedConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size,
                 num_layers,
                 batch_first=False, bias=True, return_all_layers=False, seq_len=64):
        super(AttentionWeightedConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists
        # having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.seq_len = seq_len

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[
                i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.attention = AttentionGenerator()

    def forward(self, input_tensor, first_state=None, hidden_state_ago2=None, hidden_state_ago1=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        first_state: todo
            (b, 1, h, w)
        hidden_state_ago2 & hidden_state_ago1 : todo
            (num_layers, t, b, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output_list (num_layers x b x seg_len x c x h x w)
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        # pre-process : first state
        first_state = first_state.repeat(1, length_s - 1, 1, 1, 1)
        # initialize : hidden state (ago2 and ago1)
        if hidden_state_ago2 is not None:
            pass
        else:
            hidden_state_ago2 = self._init_hidden(batch_size=input_tensor.size(0))
        if hidden_state_ago1 is not None:
            pass
        else:
            hidden_state_ago1 = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []
        cell_output_list = []

        seq_len = self.seq_len
        cur_layer_input = input_tensor

        for t in range(seq_len):
            output_inner = [] #num_layers x h
            output_cell = [] #num_layers x c
            for layer_idx in range(self.num_layers):
                # h, c = hidden_state_ago1[layer_idx]
                # h_ago2, c_ago2 = hidden_state_ago2[layer_idx]
                #                 print(cur_layer_input.shape)
                if t == 0:
                    h, c = hidden_state_ago1[layer_idx]
                    h_ago2, c_ago2 = hidden_state_ago2[layer_idx]
                    h1_ago, _ = hidden_state_ago1[0]
                    h2_ago, _ = hidden_state_ago1[1]
                    y_ago1, y_ago2 = (hidden_state_ago1[-1] == (hidden_state_ago1[-1].max(dim=-1, keepdim=True)[0]).max(dim=-1, keepdim=True)[0]).float(), (hidden_state_ago2[-1] == (hidden_state_ago2[-1].max(dim=-1, keepdim=True)[0]).max(dim=-1, keepdim=True)[0]).float()
                elif t == 1:
                    h, c = layer_output_list[t - 1][layer_idx], cell_output_list[t - 1][layer_idx]
                    h_ago2, c_ago2 = hidden_state_ago2[layer_idx]
                    h1_ago, _ = layer_output_list[t - 1][0]
                    h2_ago, _ = layer_output_list[t - 1][1]
                    y_ago1, y_ago2 = (layer_output_list[t - 1][-1] == (layer_output_list[t - 1][-1].max(dim=-1, keepdim=True)[0]).max(dim=-1, keepdim=True)[0]).float(), (hidden_state_ago2[-1] == (hidden_state_ago2[-1].max(dim=-1, keepdim=True)[0]).max(dim=-1, keepdim=True)[0]).float()
                else:
                    h, c = layer_output_list[t - 1][layer_idx], cell_output_list[t - 1][layer_idx]
                    h_ago2 = layer_output_list[t - 2][layer_idx]
                    h1_ago, _ = layer_output_list[t - 1][0]
                    h2_ago, _ = layer_output_list[t - 1][1]
                    y_ago1, y_ago2 = (layer_output_list[t - 1][-1] == (layer_output_list[t - 1][-1].max(dim=-1, keepdim=True)[0]).max(dim=-1, keepdim=True)[0]).float(), (layer_output_list[t - 2][-1] == (layer_output_list[t - 2][-1].max(dim=-1, keepdim=True)[0]).max(dim=-1, keepdim=True)[0]).float()

                if layer_idx == 0:
                    input_att = input_tensor[:, t, :, :]*self.attention(cur_layer_input[:, t, :, :], h1_ago, h2_ago)
                    input_cell = torch.cat([input_att, first_state, y_ago2, y_ago1], dim=1) # (b,<c+1+1+1>,h,w)
                    h, c = hidden_state_ago1[layer_idx]
                else:
                    input_cell = output_inner[layer_idx-1]
                    h, c = layer_output_list[t-1][layer_idx], cell_output_list[t-1][layer_idx]

                h, c = self.cell_list[layer_idx](
                    input_tensor=input_cell,
                    cur_state=[h, c])
                output_inner.append(h)
                output_cell.append(c)

            layer_output = torch.stack(output_inner, dim=0) # h=(b,c,h,w), output_inner=num_layersx(b,c,h,w), layer_output=(num_layers,b,c,h,w)
            layer_output_cell = torch.stack(output_cell, dim=0)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output) #tx(num_layer,b,c,h,w)
            cell_output_list.append(layer_output_cell)
            last_state_list.append([h, c])

        layer_output_list = torch.stack(layer_output_list, dim=2)#(num_layer,b,t,c,h,w)
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all(
                    [isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class AttentionGenerator(nn.Module):
    def __init__(self):
        super(AttentionGenerator, self).__init__()
        self.f1 = nn.Linear(28 * 28 * 64, 28 * 28 * 128)
        self.f2 = nn.Linear(28 * 28 * 16, 28 * 28 * 128)
        self.f_att = nn.Linear(28 * 28 * 128, 28 * 28)

    def forward(self, x, hidden1_pre, hidden2_pre, init=False):
        if init:
            return F.softmax(self.f_att(x+hidden1_pre+hidden2_pre), dim=-1)
        else:
            return F.softmax(self.f_att(x+self.f1(hidden1_pre)+self.f2(hidden2_pre)), dim=-1)


if __name__ == "__main__":

    x1 = torch.randn([5, 32, 60, 60]).cuda()
    x2 = torch.randn([5, 32, 60, 60]).cuda()
    x3 = torch.randn([5, 32, 60, 60]).cuda()

    cblstm = ConvBLSTM(input_size=(60, 60), input_dim=32, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True).cuda()

    x_fwd = torch.stack([x1, x2, x3], dim=1)
    x_rev = torch.stack([x3, x2, x1], dim=1)

    out = cblstm(x_fwd, x_rev)
    print(out.shape)
    out.sum().backward()
