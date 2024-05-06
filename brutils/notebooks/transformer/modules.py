from tqdm import tqdm
import copy
import math
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import brutils.notebooks.transformer as root


# TODO: have a special attention head to ensure that overlap of identical shows is complete???
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.0):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads  # head_size
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention_with_reaches(q, k, v, self.d_k, mask,
                                        self.dropout)  # [batch, num_heads, max_length, head_size]

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)  # [batch, max_length, d_model]

        output = self.out(concat)

        return output


def attention_with_reaches(q, k, v, d_k, reaches, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)  # [batch, num_heads, head_size, head_size]

    # Multiply the probabilities by the reaches and then normalize the values
    scores = scores * reaches.unsqueeze(1).unsqueeze(1)
    scores = scores - (
                (scores * .999999) * torch.eye(scores.size(-1), device=scores.device))  # A show does not affect itself

    if dropout is not None:
        scores = dropout(scores)

    # scores [batch, num_heads, max_length, max_length]
    # v [batch, num_heads, max_length, head_size]

    output = v - torch.matmul(scores, v)  # [batch, num_heads, max_length, head_size]

    # contrib [batch, max_length]
    reaches_sum = reaches.sum(dim=-1, keepdim=True)
    contrib = (reaches_sum - reaches) / (reaches_sum + 1e-9)  # Normalized via divide by reaches_sum
    contrib = contrib * (1 - reaches)  # If my reach is 1, nobody can affect me
    contrib = contrib * 100

    contrib = contrib.unsqueeze(1).unsqueeze(1)
    output = (output.transpose(2, 3) * contrib).transpose(2, 3)

    return output


class Norm(nn.Module): # Not being used
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, heads, dropout=0.0):
        super().__init__(d_model, heads)

        self.self_attn = MultiHeadAttention(d_model, heads, dropout=dropout)
        # self.norm1 = Norm(d_model)

    def forward(self, src, mask=None):
        src2 = self.self_attn(src, src, src, mask)
        src = src + self.dropout1(src2)

        # Original Transformer Encoder below, commented out because not using this
        # src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class InputDropout(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        assert p >= 0.0 and p <= 1.0
        self.p = p

    def forward(self, x, emb_size=None):  # x => [batch, max_length]
        if self.training:
            if emb_size is not None:
                emb_zero = torch.zeros(emb_size, device=x.device, dtype=x.dtype)
                choices = torch.rand(x.size()[:-1], device=x.device, dtype=x.dtype).unsqueeze(-1)
                x = torch.where(choices < self.p, emb_zero, x)
            else:
                choices = torch.rand(x.shape, device=x.device)
                x = torch.where(choices < self.p, 0, x)
        return x


class EncoderPreTre(nn.Module):
    def __init__(self, pretrained_array, d_model, N, heads, freeze_embeddings, network_vectors=None, num_dayparts=None):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(pretrained_array, dtype=torch.float32),
                                                  freeze=freeze_embeddings)
        self.post_embedding1 = nn.Linear(d_model, d_model, bias=False)
        if network_vectors is not None:
            self.netwoks_embed = nn.Embedding.from_pretrained(torch.tensor(network_vectors, dtype=torch.float32),
                                                              freeze=True)
        self.num_dayparts = num_dayparts

        self.layers = _get_clones(EncoderLayer(d_model, heads), N)
        self.pre_dropout = InputDropout(0.2)

    def forward(self, src, reaches, networks=None, dayparts=None):
        # TODO: The pre_dropout may not be working
        # src = self.pre_dropout(src)
        emb = self.embed(src)
        # emb = self.pre_dropout(emb, emb_size=emb.size(-1))
        x = emb
        # x = self.post_embedding1(emb)

        if networks is not None:
            networks = self.pre_dropout(networks)
            network_emb = self.netwoks_embed(networks)
            network_emb = self.pre_dropout(network_emb, emb_size=network_emb.size(-1))
            x = torch.cat([x, network_emb])

        if dayparts is not None:
            daypart_emb = torch.nn.functional.one_hot(dayparts, self.num_dayparts)
            daypart_emb = self.pre_dropout(daypart_emb, emb_size=daypart_emb.size(-1))
            x = torch.cat([x, daypart_emb])

        for layer in self.layers:
            x = layer(x, reaches)

        return emb, x


cos = nn.CosineSimilarity(dim=-1, eps=1e-6)


def similarity(a, b):
    return (cos(a, b) + 1) / 2


# vectors [batch, max_length, d_model]
# reaches [batch, max_length]
def sum_ortho_parallel_viewers(vectors, reaches):
    vectors = vectors / torch.norm(vectors, dim=-1).unsqueeze(-1)

    weighted_vectors = vectors * reaches.unsqueeze(-1)
    total_sum = torch.sum(weighted_vectors, dim=1).repeat(1, vectors.size(1)).view(vectors.size())

    partial_sum = total_sum - weighted_vectors
    partial_sum = partial_sum / (torch.norm(partial_sum, dim=-1).unsqueeze(-1) + 1e-8)

    A = torch.sum(partial_sum * vectors, dim=-1)  # [batch,max_length]
    vo = vectors - A.unsqueeze(dim=-1) * partial_sum
    vp = vectors - vo
    vo_norm = torch.norm(vo, dim=-1).unsqueeze(dim=-1)
    vp_norm = torch.norm(vp, dim=-1).unsqueeze(dim=-1)
    partial_sum = partial_sum + vp
    A = torch.sum(partial_sum * vectors,
                  dim=-1)  # just for the sign, which might have changed => [male_show,female_show],[0.5,0.4]

    parallel_ratio = vp_norm / (vp_norm + vo_norm)
    orth_ratio = 1 - parallel_ratio

    parallel_reach = parallel_ratio.squeeze(-1) * reaches
    orth_reach = orth_ratio.squeeze(-1) * reaches

    positive_A_mask = (A.sign() + 1) / 2
    negative_A_mask = 1 - positive_A_mask

    result = torch.max(parallel_reach * positive_A_mask, -1).values + torch.max(parallel_reach * negative_A_mask,
                                                                                -1).values

    pre_result = result.clone()  # Need to store the result so that when modifying orth_reach, we use the same value
    result += torch.max(orth_reach, dim=-1).values * (1 - pre_result)
    orth_reach = orth_reach * (((torch.sum(orth_reach, dim=-1) - torch.max(orth_reach, dim=-1).values * (
                1 - pre_result)) / (torch.sum(orth_reach, dim=-1) + 1e-8)).unsqueeze(-1))

    for i in range(reaches.size(1)):
        result += orth_reach[:, i] * (1 - result)

    return result


class ModelCGlove(nn.Module):
    def __init__(self, pretrained_array, embedding_size=50, num_layers=1, attention_heads=5, dropout=0.0,
                 freeze_embeddings=True,
                 network_vectors=None, num_dayparts=None
                 ):
        super().__init__()
        self.encoder = EncoderPreTre(pretrained_array, embedding_size, num_layers, attention_heads, freeze_embeddings,
                                     network_vectors, num_dayparts)

    def forward(self, inputs):
        # emb [batch, max_length, embedding_size]
        emb, x = self.encoder(inputs['shows'], inputs['reaches'], networks=inputs.get('networks', None),
                              dayparts=inputs.get('dayparts', None))
        return sum_ortho_parallel_viewers(x, inputs['reaches'])


class Model:
    def __init__(self, pretrained_embeddings, **model_params):
        self.model = ModelCGlove(pretrained_embeddings, **model_params)
        if torch.cuda.is_available() and root.USE_CUDA:
            self.device = torch.device('cuda', root.GPU_NUM)
            self.model.to(self.device)
        else:
            self.device = torch.device('cpu')

        self.batch_size = None

    def convert_data(self, data, data_type='train'):
        dataset = TensorDataset(
            torch.LongTensor([item['shows'] for item in data]),
            torch.FloatTensor([item['reaches'] for item in data]),
            torch.FloatTensor([item['label'] for item in data]),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True if data_type == 'train' else False)

    def __format_inputs(self, batch):
        return {
            'shows': batch[0],
            'reaches': batch[1],
            'label': batch[2],
        }

    def fit(self, train_data=None, val_data=None, train_dataloader=None, val_dataloader=None,
            batch_size=128, epochs=10, optim=torch.optim.Adagrad, lr=0.001, loss_fn=None
            ):

        self.batch_size = batch_size

        if train_dataloader is None:
            train_dataloader = self.convert_data(train_data, data_type='train')

        if val_dataloader is None and val_data is not None:
            val_dataloader = self.convert_data(val_data, data_type='val')

        optimizer = optim(self.model.parameters(), lr=lr)

        results = []
        # liveloss = PlotLosses()
        #         liveloss = PlotLosses(outputs=[BokehPlot()]) # not working for some reason
        for epoch in range(epochs):
            self.model.train()

            total_loss = 0.0
            total_l2_reg, num_timesteps = 0.0, 0.0
            with tqdm(train_dataloader, desc="Iteration") as t2:
                for step, batch in enumerate(t2):
                    optimizer.zero_grad()

                    batch = tuple(t.to(self.device) for t in batch)
                    inputs = self.__format_inputs(batch)
                    output = self.model(inputs)
                    loss = loss_fn(inputs['label'], output)
                    total_loss += loss.item()
                    num_timesteps += 1

                    loss.backward()
                    optimizer.step()
                    t2.set_postfix(train_loss=total_loss / num_timesteps)

            total_loss /= num_timesteps
            current_result = {'epoch': epoch + 1}
            current_plot = {}
            current_result['loss'] = total_loss

            train_metrics = self.evaluate(dataloader=train_dataloader, loss_fn=loss_fn)
            for i, metric in enumerate(['loss', 'r2', 'err', 'err_weight']):
                current_result['train_' + metric] = current_plot[metric] = train_metrics[i]

            if val_dataloader is not None:
                val_metrics = self.evaluate(dataloader=val_dataloader, loss_fn=loss_fn)
                for i, metric in enumerate(['loss', 'r2', 'err', 'err_weight']):
                    current_result['val_' + metric] = current_plot['val_' + metric] = val_metrics[i]

            # liveloss.update(current_plot)
            # liveloss.send()
            # time.sleep(1)
            results.append(current_result)

        #         clear_output(wait=True)
        return pd.DataFrame(results)

    def evaluate(self, dataloader=None, loss_fn=None):
        total_loss, num_timesteps = 0.0, 0

        unexp_var = 0
        tot_var = 0
        tot_err = 0
        tot_l1_norm = 0

        all_labels = []
        all_preds = []

        self.model.eval()
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = self.__format_inputs(batch)
                output = self.model(inputs)
                loss = loss_fn(inputs['label'], output)
                all_preds += list(output.detach().cpu().numpy())
                all_labels += list(inputs['label'].detach().cpu().numpy())

                total_loss += loss.item()
                num_timesteps += 1

                unexp_var += torch.sum((inputs['label'] - output) ** 2).item()
                tot_var += torch.sum(inputs['label'] ** 2).item()
                tot_err += torch.sum(torch.abs(inputs['label'] - output)).item()
                tot_l1_norm += torch.sum(torch.abs(inputs['label'])).item()

        total_loss = total_loss / num_timesteps
        r2 = 1 - unexp_var / tot_var
        err = tot_err / len(dataloader)
        err_weight = tot_err / tot_l1_norm

        return total_loss, r2, err, err_weight

    def predict(self, data=None, dataloader=None):
        if dataloader is None:
            dataloader = self.convert_data(data, data_type='val')

        preds = []

        self.model.eval()
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = self.__format_inputs(batch)
                output = self.model(inputs)
                preds += list(output.detach().cpu().numpy())

        return np.array(preds)
