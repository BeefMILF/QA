import torch
from torch import nn
import torch.nn.functional as F


class DrQA(nn.Module):
    def __init__(self, config, dl, device):
        super().__init__()
        self._init_configure(config, dl, device)
        self._init_trainable()

    def _init_configure(self, config, dl, device):
        self.batch_size = config['batch_size']
        self.n_layers = config['n_layers']
        self.hidden_size = config['hidden_size']
        self.embedding_dim = config['embedding_dim']
        self.par_embedding = config['paragraph_embedding']
        self.bidirectional = config['bidirectional']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.div = (1 + self.bidirectional) * self.n_layers

        self.encode_q = config['encode_q']
        self.pooler = config['pooler']['mode']
        self.kernel_size = config['pooler']['kernel_size']
        self.stride = config['pooler']['stride']

        self.pretrained = config['pretrained']

        self.dl = dl
        self.device = device

    def _init_trainable(self):
        # word emb
        self.word_emb = nn.Embedding.from_pretrained(self.pretrained, freeze=True)

        self.LSTM_par = nn.LSTM(input_size=self.par_embedding,
                                hidden_size=self.hidden_size // self.div,
                                num_layers=self.n_layers,
                                bidirectional=self.bidirectional,
                                batch_first=True,
                                dropout=self.dropout)

        self.LSTM_query = nn.LSTM(input_size=self.embedding_dim,
                                  hidden_size=self.hidden_size // self.div,
                                  num_layers=self.n_layers,
                                  bidirectional=self.bidirectional,
                                  batch_first=True,
                                  dropout=self.dropout)

        self.alpha_layer = nn.Linear(self.embedding_dim, 1)

        if self.encode_q:
            self._weights = nn.Linear(self.hidden_size // self.div * 2, 1)

        self.flatten = nn.Flatten()

        # after concat question & passage we have too much parameters for fc layer
        # we can run through max pooling layer
        fc_in = 4 * self.hidden_size if not self.encode_q else 2 * (self.hidden_size + self.hidden_size // self.div)
        if self.pooler:
            self.pooler = nn.MaxPool1d(self.kernel_size, self.stride)
            fc_in = int((fc_in - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
        else:
            self.pooler = None

        self.fc1 = nn.Linear(fc_in, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.fc2 = nn.Linear(self.hidden_size, 2)

    ## Paragraph part
    # Ner
    def get_ner_ind(self, word_ind: int):
        out = torch.zeros(len(self.dl.ner_voc))
        ind = self.dl.ind2ner[word_ind]  # default 0
        out[ind] = 1
        return out

    def get_ner_4_pars(self, batch):
        # batch - (bs, seq_len)
        n, m = batch.shape
        result = torch.zeros((n, m, len(self.dl.ner_voc)))
        for i in range(n):
            for j in range(m):
                result[i, j, :] = self.get_ner_ind(batch[i, j].item())
        return result.to(self.device)

    # Pos-tags
    def get_pos_ind(self, word_ind: int):
        out = torch.zeros(len(self.dl.pos_voc))
        ind = self.dl.ind2pos[word_ind]
        out[ind] = 1
        return out

    def get_pos_4_pars(self, batch):
        # batch - (bs, seq_len)
        n, m = batch.shape
        result = torch.zeros((n, m, len(self.dl.pos_voc)))
        for i in range(n):
            for j in range(m):
                result[i, j, :] = self.get_pos_ind(batch[i, j].item())
        return result.to(self.device)

    # Match check
    def check_match(self, word_ind, query):
        return torch.tensor(word_ind in query).long()

    def get_match_4_pars(self, batch, query):
        n, m = batch.shape
        result = torch.zeros((n, m, 1))
        for i in range(n):
            for j in range(m):
                result[i, j] = self.check_match(batch[i, j], query[i])
        return result.to(self.device)

    # Soft-attention
    def alpha(self, vec):
        out = self.alpha_layer(vec.float())
        return F.relu(out)

    def f_align(self, p, query):
        # p - (emb_dim)
        # query - (seq_len, emb_dim) 
        p_a = torch.exp(self.alpha(p))  # (1)
        q_a = self.alpha(query).squeeze(1)  # (seq_len)

        summ = (p_a * q_a).sum(0)  # (1)
        out = ((p_a * q_a)[:, None] * query / summ).sum(0)  # (emb_dim)
        return out

    def f_align_4_pars(self, batch, query):
        # batch - (bs, seq_len, emb_dim)
        # query - (bs, seq_len, emb_dim) 
        n, m, _ = batch.shape
        result = torch.zeros((n, m, self.embedding_dim))
        for i in range(n):
            for j in range(m):
                result[i, j] = self.f_align(batch[i, j], query[i])
        return result.to(self.device)

    ## Query part
    def weights(self, query):
        # query - (n_layers, hid_dim * 2)
        return self._weights(query)

    def q_importance(self, query):
        # query - (n_layers, hid_dim * 2)
        q_w = torch.exp(self.weights(query))  # (n_layers, 1)
        summ = q_w.sum(0)  # (1)
        out = (q_w * query / summ).sum(0)  # (hid_dim * 2)
        return out

    def q_encode(self, query):
        # query - (bs, n_layers, hid_dim * 2) 
        n, _, _ = query.shape
        result = torch.zeros((n, self.hidden_size // self.div * 2))
        for i in range(n):
            result[i] = self.q_importance(query[0])
        return result.to(self.device)

    def forward(self, batch):
        # pars - (seq_len, bs)
        # query - (seq_len, bs)
        # query and pars have different seq_len 

        pars = batch.context_word.T.to(self.device)  # (bs, seq_len)
        query = batch.question_word.T.to(self.device)  # (bs, seq_len)

        pars_emb = self.word_emb(pars)  # (bs, seq_len, emb_dim)
        query_emb = self.word_emb(query)  # (bs, seq_len, emb_dim)

        ## Paragraph encoding
        pars_ner = self.get_ner_4_pars(pars)  # (bs, seq_len, ner_voc)
        pars_pos = self.get_pos_4_pars(pars)  # (bs, seq_len, pos_voc)

        pars_match = self.get_match_4_pars(pars, query)  # (bs, seq_len, 1)

        pars_soft_attn = self.f_align_4_pars(pars_emb, query_emb)

        conc_pars = torch.cat((pars_emb, pars_ner, pars_pos, pars_soft_attn, pars_match), 2)  # (bs, seq_len, 637)
        # 637(paragraph features) = len(dl.ner_voc) + len(dl.pos_voc) + 300 * 2 + 1 

        par_lstm_out = self.LSTM_par(conc_pars)[1]  # tuple(h_n, c_n) of (n_layers, bs, hid_dim)
        par_lstm_out = torch.cat((par_lstm_out[0].permute(1, 0, 2).reshape(-1, self.hidden_size),
                                  par_lstm_out[1].permute(1, 0, 2).reshape(-1, self.hidden_size)), 1)

        if not self.encode_q:
            query_lstm_out = self.LSTM_query(query_emb)[1]  # tuple(h_n, c_n) of (n_layers, bs, hid_dim)
            query_lstm_out = torch.cat((query_lstm_out[0].permute(1, 0, 2).reshape(-1, self.hidden_size),
                                        query_lstm_out[1].permute(1, 0, 2).reshape(-1, self.hidden_size)), 1)
            # par_lstm_out - (bs, n_layers * hid_dim)
            # par_lstm_out - (bs, n_layers * hid_dim)
        else:
            query_lstm_out = self.LSTM_query(query_emb)[1]  # tuple(h_n, c_n) of (n_layers, bs, hid_dim)
            query_lstm_out = torch.cat((query_lstm_out[0].permute(1, 0, 2), query_lstm_out[1].permute(1, 0, 2)), 2)  # (bs, n_layers, hid_dim * 2)
            query_lstm_out = self.q_encode(query_lstm_out)

        res = torch.cat((par_lstm_out, query_lstm_out), 1)  # (bs, n_layers * hid_dim * 2)

        if self.pooler is not None:
            res = self.pooler(res.unsqueeze(1))  # 3 dimensions required for pooler
            res = res.squeeze(1)

        res = self.fc1(self.flatten(res))
        res = self.dropout(res)
        res = self.fc2(res)

        return res


