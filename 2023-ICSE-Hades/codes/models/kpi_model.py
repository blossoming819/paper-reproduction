import logging

import torch
from torch import nn
from torch.autograd import Variable
from models.utils import Decoder


# Trim the excess right padding from the output of Causal Convolution to maintain temporal alignment between input and output.
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size  # the padding length that needs to be trimmed

    def forward(self, x):
        # Ensure the tensor continuous to prevent the errors caused by calling view after transpose or permute
        return x[:, :, :-self.chomp_size].contiguous()


# input: [batch_size, T, input_size]
# output: if existing the MaxPooling layer, [batch_size, T, 1]; else [batch_size, T, output_size]
class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes, dilation=2, device="cpu", dropout=0, pooling=True,
                 **kwargs):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(len(kernel_sizes)):
            dilation_size = dilation ** i  # The dilation coefficient growth exponentially.
            kernel_size = kernel_sizes[i]  # Multiple kernel
            padding = (kernel_size - 1) * dilation_size  # The padding length is d * (k - 1)
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels,
                          out_channels,
                          kernel_size,
                          stride=1,
                          dilation=dilation_size,
                          padding=padding),
                nn.ReLU(),
                Chomp1d(padding),
                nn.Dropout(dropout)]

        self.network = nn.Sequential(*layers)

        self.pooling = pooling
        if self.pooling:
            self.maxpool = nn.MaxPool1d(num_channels[-1]).to(device)  # global max pooling
        self.network.to(device)

    def forward(self, x):  # [batch_size, T, input_size]
        x = x.permute(0, 2, 1)  # [batch_size, input_size, T]
        out = self.network(x)  # [batch_size, out_dim, T]
        out = out.permute(0, 2, 1)  # [batch_size, T, out_dim]
        if self.pooling:
            return self.maxpool(out)  # [batch_size, T, 1]
        else:
            return out

# input: [batch_size, T, input_size]
# output: [batch_size, T, temporal_dim]
class TemporalEncoder(nn.Module):
    def __init__(self, device, input_size, **kwargs):
        super(TemporalEncoder, self).__init__()
        hidden_sizes = kwargs["temporal_hidden_sizes"]  # default [64, 4]
        kernel_sizes = kwargs["temporal_kernel_sizes"]  # default [2, 2]
        dropout = kwargs["temporal_dropout"]  # default 0
        pooling = kwargs["pooling"]  # default True
        self.temporal_dim = hidden_sizes[-1]

        self.temporal_dim = 1 if pooling else hidden_sizes[-1]
        self.net = ConvNet(input_size, hidden_sizes, kernel_sizes, device=device, dropout=dropout, pooling=pooling)

    # [batch_size, T, input_size] --> [batch_size, T, temporal_dim]
    def forward(self, x: torch.Tensor):
        x = x.type("torch.FloatTensor").to(x.device)
        return self.net(x)

# input: [batch_size, T, input_size]
# output: [batch_size, T, var_num or metric_num]
class InnerEncoder(nn.Module):
    def __init__(self, input_size, device, **kwargs):
        super(InnerEncoder, self).__init__()
        temporal_dims = kwargs["inner_hidden_sizes"]  # [256, 256, 256]
        kernel_sizes = kwargs["inner_kernel_sizes"]  # [3, 3, 3]
        dropout = kwargs["inner_dropout"]  # 0.5

        assert len(temporal_dims) == len(kernel_sizes)
        temporal_dims[-1] = kwargs["hidden_size"]
        self.net = ConvNet(input_size, temporal_dims, kernel_sizes, device=device, dropout=dropout, pooling=False)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class KpiEncoder(nn.Module):
    def __init__(self, var_nums, device, **kwargs):
        super(KpiEncoder, self).__init__()
        self.kpi_type = kwargs["kpi_architect"]
        self.var_nums = var_nums
        self.metric_num = sum(var_nums)
        self.group_num = len(var_nums)

        self.window_size = kwargs["window_size"]  # 10

        if self.kpi_type == "by_aspect":
            self.t_encoders = [TemporalEncoder(device, input_size=var, **kwargs) for var in self.var_nums]
            self.temporal_dim = self.t_encoders[0].temporal_dim
            self.i_encoder = InnerEncoder(self.group_num * self.temporal_dim, device, **kwargs)

        elif self.kpi_type == "by_metric":
            self.t_encoders = [TemporalEncoder(device, 1, **kwargs) for _ in range(self.metric_num)]
            self.i_encoder = InnerEncoder(self.metric_num, device, **kwargs)
            self.temporal_dim = self.t_encoders[0].temporal_dim

        else:
            raise ValueError("Unrecognized Kpi Architect Type {}!".format(self.kpi_type))

    def forward(self, ts):  # ts group list
        batch_size = ts[0].size(0)
        d = ts[0].device

        if self.kpi_type == "by_aspect":
            group_encodings = []
            for i, group in enumerate(ts): # group: [batch_size, var_num, T]
                aspect_input = group.permute(0, 2, 1) # [batch_size, T, var_num]
                encoded_group = self.t_encoders[i].forward(aspect_input)  # [batch_size, T, temporal_dim]
                group_encodings.append(encoded_group)
            inner_input = torch.cat(group_encodings, dim=-1)  # [batch_size, T, temporal_dim * group_num]
            return self.i_encoder(inner_input)

        elif self.kpi_type == "by_metric":
            m = 0
            encoded_metric = Variable(
                torch.zeros(batch_size, self.window_size, self.metric_num, self.temporal_dim, device=d))
            for i, group in enumerate(ts):  # group: [batch_size, var_num, T]
                group = group.permute(0, 2, 1) # modify [batch_size, T, var_num]
                for j in range(self.var_nums[i]):
                    # metric_input = group[:, j:j + 1, :].permute(0, 2, 1)  # [batch_size, 1, T] --> [batch_size, T, 1]
                    metric_input = group[:, :, j:j + 1]  # [batch_size, T, 1]
                    # encoded_metric[m] = self.t_encoders[m].forward(metric_input)
                    encoded_metric[:, :, m, :] = self.t_encoders[m].forward(metric_input) # modify
                    m += 1
            # encoded_metric = torch.mean(encoded_metric,
            #                             dim=-1).squeeze()  # [metric_num, batch_size, T, temporal_dim] --> [metric_num, batch_size, T]
            inner_input = torch.mean(encoded_metric,
                                        dim=-1).squeeze()  # [batch_size, T, metric_num, temporal_dim] --> [batch_size, T, metric_num] modify
            # logging.info(str(encoded_metric.shape))
            # inner_input = encoded_metric.permute(1, 2, 0)  # --> [batch_size, T, metric_num]
            return self.i_encoder(inner_input)  # [batch_size, T, hidden_size]


from torch.nn.functional import softmax as sf


class KpiModel(nn.Module):
    def __init__(self, var_nums, device, **kwargs):
        super(KpiModel, self).__init__()  # init BaseModel
        self.var_nums = var_nums
        self.encoder = KpiEncoder(var_nums, device, **kwargs)
        self.decoder = Decoder(kwargs["hidden_size"], kwargs["window_size"], **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_dict, flag=False):
        y = input_dict["label"].long().view(-1)  # [batch_size, ]

        kpi_re = self.encoder(input_dict["kpi_features"])
        logits = self.decoder(kpi_re)  # [batch_size, 2]
        if y.size(0) == 1: logits = logits.unsqueeze(0)
        if flag:
            y_pred = logits.detach().cpu().numpy().argmax(axis=1)
            conf = sf(logits.detach().cpu(), dim=1).numpy().max(axis=1)  # [bz, 1]
            return {"y_pred": y_pred, "conf": conf}

        loss = self.criterion(logits, y)
        return {"loss": loss}
