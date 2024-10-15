from scipy.special import expit
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import json
import time
import logging 
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(message)s',  
)

def get_arguments():
    def get_params(filePath):
        with open(filePath, 'r') as f:
            params = json.load(f)
        f.close()
        return params
    train_params = get_params("./params/train_params.json")
    leak_gan_params = get_params("./params/leak_gan_params.json")
    dis_data_params = get_params("./params/dis_data_params.json")
    real_data_params = get_params("./params/real_data_params.json")
    return {
        "train_params": train_params,
        "leak_gan_params": leak_gan_params,
        "dis_data_params": dis_data_params,
        "real_data_params" : real_data_params,
    }

def init_vars(generator, discriminator, use_cuda=False):
    h_w_t, c_w_t, h_m_t, c_m_t = generator.init_hidden() 
    x_t = nn.init.constant_(torch.Tensor(generator.worker.batch_size), discriminator.start_token).long()
    variables_ = [h_w_t, c_w_t, h_m_t, c_m_t, x_t]
    vs = []
    if use_cuda:
        for var in variables_:
            var = var.cuda(non_blocking=True)
            vs.append(var)
    else:
        vs = variables_
    return vs

def recurrent_func(f_type = "pre"):
    """
    There are 3 types of recurrent function:
        1. pre = pretrain
        2. adv = adversarial train
        3. rollout = rollout for evaluate reward

    Each kind of training has its own function
    """
    if f_type == "pre":
        def func(model_dict, real_data, use_cuda, temperature = 1.0):
            '''
            discriminator is detached.
            real_data shape: [batch_size, seq_length]
            '''
            generator = model_dict["generator"]
            discriminator = model_dict["discriminator"]
            batch_size = generator.worker.batch_size
            seq_len = discriminator.seq_len
            vocab_size = discriminator.vocab_size
            h_w_t, c_w_t, h_m_t, c_m_t, x_t = init_vars(generator, discriminator, use_cuda)

            t = 0
            cur_sen = nn.init.constant_(torch.zeros(batch_size, seq_len), vocab_size).long()
            if use_cuda: cur_sen = cur_sen.cuda(non_blocking = True)
            feature_list = []
            real_goal_list = []
            prediction_list = []
            while t < seq_len:
                f_t = discriminator(cur_sen)["feature"]
                # f_t = f_t.detach() -> Not requried; discriminator in eval mode.
                # f_t = nn.init.constant_(torch.zeros(*f_t.shape), 5.0 * 1.1 * t)
                feature_list.append(f_t)
                x_t, h_m_t, c_m_t, h_w_t, c_w_t, real_goal, probs, t_ = generator(x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, t, temperature)
                real_goal_list.append(real_goal)
                prediction_list.append(probs)
                cur_sen = real_data[:,:(t+1)]
                cur_sen = cur_sen.contiguous()
                cur_sen = F.pad(cur_sen.view(-1, t + 1), (0, seq_len - (t+1)), value = vocab_size)
                if use_cuda: cur_sen = cur_sen.cuda(non_blocking = True)
                t = t_

            # If not reduced size would be: [batch_size, seq_length, f_t's dim]
            real_goal_var = torch.stack(real_goal_list).permute(1,0,2) 
            prediction_var = torch.stack(prediction_list).permute(1,0,2) 
            feature_list = torch.stack(feature_list).permute(1, 0, 2)
            results = {
                        "real_goal": real_goal_var,
                        "prediction": prediction_var, 
                        "feature_list": feature_list
                    }
            for result in results.values():
                if not result.is_contiguous(): result = result.contiguous()
            return results
        return func
 
    elif f_type == "adv":
        def func(model_dict, use_cuda=False, temperature = 1.0):
            generator = model_dict["generator"]
            discriminator = model_dict["discriminator"]
            batch_size = generator.worker.batch_size
            seq_len = discriminator.seq_len
            vocab_size = discriminator.vocab_size
            h_w_t, c_w_t, h_m_t, c_m_t, x_t = init_vars(generator, discriminator, use_cuda)
            t = 0
            cur_sen = nn.init.constant_(torch.zeros(batch_size, seq_len), vocab_size).long()
            if use_cuda: cur_sen = cur_sen.cuda(non_blocking = True)
            feature_list, prediction_list, real_goal_list, gen_token_list = [], [], [], []
            while t < seq_len:
                f_t = discriminator(cur_sen)["feature"]
                feature_list.append(f_t)
                x_t, h_m_t, c_m_t, h_w_t, c_w_t, real_goal, probs, t_ = generator(x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, t, temperature)
                gen_token_list.append(x_t)
                real_goal_list.append(real_goal)
                prediction_list.append(probs)
                cur_sen = torch.stack(gen_token_list).permute(1,0)
                cur_sen = F.pad(cur_sen, (0, seq_len - t), value=vocab_size)
                if use_cuda: cur_sen = cur_sen.cuda(non_blocking = True)
                t = t_

            real_goal_var = torch.stack(real_goal_list).permute(1,0,2)
            prediction_var = torch.stack(prediction_list).permute(1,0,2)
            feature_list = torch.stack(feature_list).permute(1, 0, 2)
            gen_token_var = torch.stack(gen_token_list).permute(1,0)
            results = {
                "real_goal": real_goal_var,
                "prediction": prediction_var,
                "gen_token": gen_token_var,
                "feature_list": feature_list
            }
            for result in results.values():
                if not result.is_contiguous(): result = result.contiguous()
            return results
        return func
        
    elif f_type == "gen":
        '''
        Don't modify generator as this is only used for sampling.
        Don't modify discriminator as this is only used for sampling.
        '''
        def func(model_dict, use_cuda=False, temperature=1.0):
            generator = model_dict["generator"]
            discriminator = model_dict["discriminator"]
            batch_size = generator.worker.batch_size
            seq_len = discriminator.seq_len
            vocab_size = discriminator.vocab_size
            h_w_t, c_w_t, h_m_t, c_m_t, x_t = init_vars(generator, discriminator, use_cuda)
            t = 0
            cur_sen = nn.init.constant_(torch.zeros(batch_size, seq_len), vocab_size).long()
            if use_cuda: cur_sen = cur_sen.cuda(non_blocking = True)
            gen_token_list = []
            while t < seq_len:
                f_t = discriminator(cur_sen)["feature"]
                x_t, h_m_t, c_m_t, h_w_t, c_w_t, real_goal, probs, t_ = generator(x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, t, temperature)
                gen_token_list.append(x_t)
                cur_sen = torch.stack(gen_token_list).permute(1, 0)
                cur_sen = F.pad(cur_sen, (0, seq_len - t), value=vocab_size)
                if use_cuda: cur_sen = cur_sen.cuda(non_blocking = True)
                t = t_
            gen_token = torch.stack(gen_token_list).permute(1,0)
            return gen_token
        return func
    else:
        raise("Invalid funnction type")
    
def get_sample(model_dict, use_cuda=False, temperature=1.0):
    return recurrent_func("gen")(model_dict, use_cuda, temperature)

def get_rewards(model_dict, input_x, use_cuda=False, temperature=1.0, delta=12.0):
    discriminator = model_dict["discriminator"]
    pred = discriminator(input_x)["pred"]
    rewards_type1 = F.softmax(pred, dim = 1)[:,:1]
    # rewards_type2 = rescale(pred[:,:1], delta) / rollout_num
    return rewards_type1

def rescale(rewards, delta=12.0):
    '''
    delta controls how extreme lower and upper probabilities are.
    '''
    r = np.array(rewards)
    _, batch_size = r.shape
    order = np.argsort(r)
    rank = np.argsort(order)
    rank = batch_size - rank
    rescaled_rewards = expit(delta*(0.5 - rank/batch_size))
    rescaled_rewards = np.transpose(rescaled_rewards) # To make batch_size the first dimension.
    return Variable(torch.from_numpy(rescaled_rewards)).float()

def one_hot(x, vocab_size, use_cuda = False):
    batch_size, seq_len = x.size()
    out = torch.zeros(batch_size * seq_len, vocab_size, device=x.device)
    x = x.contiguous()
    x = x.view(-1, 1)
    if (x.data < vocab_size).all() == 0:
        for i, d in enumerate(x.data):
            if x[i].item() > vocab_size - 1 :
                x[i] = 0
    out = out.scatter_(1, x.data, 1.0) #setting particular values of a tensor at the provided indices, one hot vector at positions where there is word
    out = out.view(batch_size, seq_len, vocab_size)
    out = Variable(out)
    if use_cuda:
        out = out.cuda(non_blocking=True)
    return out

def loss_func(f_type="pre_worker"):
    """
    5 kind of loss function: pre_worker, pre_manager, adv_worker, adv_manager, dis
    """
    if f_type == "pre_worker":
        def func(real_data, prediction, vocab_size, use_cuda):
            # logging.debug(prediction)
            loss = -torch.mean(torch.sum(one_hot(real_data, vocab_size, use_cuda) * torch.log(prediction), dim=2))
            return loss
        return func
    elif f_type == "adv_worker":
        def func(gen_token, prediction, rewards, vocab_size, use_cuda):
            rewards = rewards.to(gen_token.device)
            log_loss = -torch.sum(one_hot(gen_token, vocab_size, use_cuda) *  torch.log(prediction), dim=2)
            loss = torch.mean(rewards * log_loss)
            return loss
        return func
    else:
        raise("Invalid loss function type")