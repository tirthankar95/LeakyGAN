from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import logging

# A truncated distribution has its domain (the x-values) restricted to a certain range of values. For example, you might restrict your x-values to between 0 and 100, written in math terminology as {0 > x > 100}. There are several types of truncated distributions:
def truncated_normal(shape, lower=-0.2, upper=0.2):
    size = 1
    for dim in shape:
        size *= dim
    w_truncated = truncnorm.rvs(lower, upper, size=size)
    w_truncated = torch.from_numpy(w_truncated).float()
    w_truncated = w_truncated.view(shape)
    return w_truncated

class Manager(nn.Module):
    def __init__(self, batch_size, hidden_dim, goal_out_size):
        super(Manager, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.goal_out_size = goal_out_size
        self.recurrent_unit = nn.LSTMCell(
            self.goal_out_size, #input size
            self.hidden_dim #hidden size
        )
        self.fc = nn.Linear(
            self.hidden_dim, #in_features
            self.goal_out_size #out_features
        )
        self.goal_init = nn.Parameter(torch.zeros(self.batch_size, self.goal_out_size))
        self.last_goal_wts = torch.ones(self.batch_size, self.goal_out_size) * (1/2)
        self.last_goal = torch.zeros(self.batch_size, self.goal_out_size, dtype = torch.float32, requires_grad = True)
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)
        self.goal_init.data = truncated_normal(self.goal_init.data.shape)

    def forward(self, f_t, h_m_t, c_m_t):
        h_m_tp1, c_m_tp1 = self.recurrent_unit(f_t, (h_m_t, c_m_t))
        sub_goal = self.fc(h_m_tp1)
        sub_goal = torch.renorm(sub_goal, 2, 0, 1.0) #Returns a tensor where each sub-tensor of input along dimension dim is normalized such that the p-norm of the sub-tensor is lower than the value maxnorm
        return sub_goal, h_m_tp1, c_m_tp1

class Worker(nn.Module):
    def __init__(self, batch_size, vocab_size, embed_dim, hidden_dim, 
                    goal_out_size, goal_size):
        super(Worker, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.goal_out_size = goal_out_size
        self.goal_size = goal_size
        self.emb = nn.Embedding(self.vocab_size, self.embed_dim)
        self.recurrent_unit = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.goal_size * self.vocab_size)
        self.goal_change = nn.Parameter(torch.zeros(self.goal_out_size, self.goal_size))
        self._init_params()
        
    def _init_params(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)

    def forward(self, x_t, h_w_t, c_w_t):
        """
            x_t = last word
            h_w_t = last output of LSTM in Worker
            c_w_t = last cell state of LSTM in Worker
        """
        x_t_emb = self.emb(x_t)
        h_w_tp1, c_w_tp1 = self.recurrent_unit(x_t_emb, (h_w_t, c_w_t))
        output_tp1 = self.fc(h_w_tp1)
        output_tp1 = output_tp1.view(self.batch_size, self.vocab_size, self.goal_size)
        return output_tp1, h_w_tp1, c_w_tp1

class Generator(nn.Module):
    def __init__(self, worker_params, manager_params, step_size):
        super(Generator, self).__init__()
        self.step_size = step_size
        self.worker = Worker(**worker_params)
        self.manager = Manager(**manager_params)
        
    def init_hidden(self):
        batch_size, goal_out_size = self.manager.batch_size, self.manager.goal_out_size
        self.manager.last_goal = torch.zeros(batch_size, goal_out_size, dtype = torch.float32, requires_grad = True)
        h_w = Variable(torch.zeros(self.worker.batch_size, self.worker.hidden_dim))
        c_w = Variable(torch.zeros(self.worker.batch_size, self.worker.hidden_dim))
        h_m = Variable(torch.zeros(self.worker.batch_size, self.worker.hidden_dim))
        c_m = Variable(torch.zeros(self.worker.batch_size, self.worker.hidden_dim))
        return h_w, c_w, h_m, c_m

    def forward(self, x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, t, temperature):
        sub_goal, h_m_tp1, c_m_tp1 = self.manager(f_t, h_m_t, c_m_t)
        output, h_w_tp1, c_w_tp1 = self.worker(x_t, h_w_t, c_w_t)
        sub_goal = F.normalize(sub_goal, dim = 1)
        self.manager.last_goal = self.manager.last_goal.to(sub_goal.device)
        self.worker.goal_change = self.worker.goal_change.to(sub_goal.device)
        self.manager.last_goal_wts = self.manager.last_goal_wts.to(sub_goal.device)
        self.manager.last_goal = self.manager.last_goal_wts * self.manager.last_goal + sub_goal
        w_t = torch.matmul(sub_goal, self.worker.goal_change)
        w_t = torch.renorm(w_t, 2, 0, 1.0)
        w_t = torch.unsqueeze(w_t, -1)
        logits = torch.squeeze(torch.matmul(output, w_t))
        probs = F.softmax(logits/temperature, dim = 1)
        x_tp1 = Categorical(probs).sample()
        return x_tp1, h_m_tp1, c_m_tp1, h_w_tp1, c_w_tp1, sub_goal, probs, t + 1
    
    def get_model_wts(self, is_trainable = True):
        if is_trainable:
            return sum([p.numel() for p in self.parameters() if p.requires_grad == True ])
        return sum([p.numel() for p in self.parameters()])