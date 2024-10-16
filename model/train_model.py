import pickle as pkl
import numpy as np
import json 
import glob
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
torch.autograd.set_detect_anomaly(True)

from data_iter import real_data_loader, dis_data_loader
from model.utils import recurrent_func, loss_func, get_sample, get_rewards, get_arguments
from model.Discriminator import Discriminator
from model.Generator import Generator
import model.Bleu as Bleu

# Global param value.
param_dict, step_size = None, None
LOG_MOD = 10

# List of models
def prepare_model_dict(use_cuda=False):
    param = param_dict["leak_gan_params"]
    discriminator_params = param["discriminator_params"]
    generator_params = param["generator_params"]
    worker_params = generator_params["worker_params"]
    manager_params = generator_params["manager_params"]
    discriminator_params["goal_out_size"] = sum(
        discriminator_params["num_filters"]
    )
    worker_params["goal_out_size"] = discriminator_params["goal_out_size"]
    manager_params["goal_out_size"] = discriminator_params["goal_out_size"]
    discriminator = Discriminator(**discriminator_params)
    generator = Generator(worker_params, manager_params,
                          generator_params["step_size"])
    if use_cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    model_dict = {"generator": generator, "discriminator": discriminator}
    return model_dict

# List of optimizers
def prepare_optimizer_dict(model_dict, lr_dict): # lr_dict = learning rate 
    generator = model_dict["generator"]
    discriminator = model_dict["discriminator"]
    worker = generator.worker
    manager = generator.manager
    m_lr = lr_dict["manager"]
    w_lr = lr_dict["worker"]
    d_lr = lr_dict["discriminator"]
    w_optimizer = optim.Adam(worker.parameters(), lr = w_lr)
    m_optimizer = optim.Adam(manager.parameters(), lr = m_lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr = d_lr)
    return {"worker": w_optimizer, "manager": m_optimizer,
            "discriminator": d_optimizer}

# List of Learning rate Schedulers
def prepare_scheduler_dict(optmizer_dict, step_size=200, gamma=0.99):
    w_optimizer = optmizer_dict["worker"]
    m_optimizer = optmizer_dict["manager"]
    d_optimizer = optmizer_dict["discriminator"]
    w_scheduler = optim.lr_scheduler.StepLR(w_optimizer, step_size = step_size, gamma = gamma)
    m_scheduler = optim.lr_scheduler.StepLR(m_optimizer, step_size = step_size, gamma = gamma)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size = step_size, gamma = gamma)
    return {"worker": w_scheduler, "manager": m_scheduler,
            "discriminator": d_scheduler}

# Pretraining the Generator
def pretrain_generator(model_dict, optimizer_dict, scheduler_dict, dataloader, \
                       vocab_size, use_cuda = False):
    # get the models of generator
    generator = model_dict["generator"]
    worker = generator.worker
    manager = generator.manager
    # get the optimizers
    m_optimizer = optimizer_dict["manager"]
    w_optimizer = optimizer_dict["worker"]
    m_lr_scheduler = scheduler_dict["manager"]
    w_lr_scheduler = scheduler_dict["worker"]
    m_loss, w_loss = 0, 0
    """
     Perform pretrain step for real data
    """
    for i, sample in enumerate(dataloader):
        if use_cuda: sample = sample.cuda(non_blocking = True)
        # Calculate pretrain loss
        param = param_dict["leak_gan_params"]
        batch_size, seq_length = param["generator_params"]["manager_params"]["batch_size"],\
                                 param["discriminator_params"]["seq_len"]
        if (sample.size() == torch.Size([batch_size, seq_length])): #sometimes smaller than 64 (16) is passed, so this if statement disables it
            '''
            Graph is released after m_optimizer.step() 
            so it's calculated again.
            '''
            # Worker
            w_optimizer.zero_grad()
            pre_rets = recurrent_func("pre")(model_dict, sample, use_cuda)
            prediction = pre_rets["prediction"].squeeze()
            w_loss = loss_func("pre_worker")(sample, prediction, vocab_size, use_cuda)
            w_loss.backward()
            w_optimizer.step()
            m_lr_scheduler.step()
            w_lr_scheduler.step()
            if i % LOG_MOD == 0:
                logging.debug("Pre-Manager Loss: {:.5f}, Pre-Worker Loss: {:.5f}\n".format(m_loss, w_loss))
    """
    Update model_dict, optimizer_dict, and scheduler_dict
    """
    generator.woroker = worker
    generator.manager = manager
    model_dict["generator"] = generator
    optimizer_dict["manager"] = m_optimizer
    optimizer_dict["worker"] = w_optimizer
    scheduler_dict["manager"] = m_lr_scheduler
    scheduler_dict["worker"] = w_lr_scheduler
    return model_dict, optimizer_dict, scheduler_dict

def generate_samples(model_dict, negative_file, batch_size,
                     use_cuda=False, temperature=1.0):
    # [True|False] ~ [Training Mode| Eval Mode]
    dis_state, gen_state = model_dict["discriminator"].training, \
                           model_dict["generator"].training
    if dis_state: model_dict["discriminator"] = model_dict["discriminator"].eval()
    if gen_state: model_dict["generator"] = model_dict["generator"].eval()
    neg_data = []
    for _ in range(batch_size):
        sample = get_sample(model_dict, use_cuda, temperature)
        sample = sample.cpu()
        neg_data.append(sample.data.numpy())
    neg_data = np.concatenate(neg_data, axis=0)
    np.save(negative_file, neg_data)
    if dis_state: model_dict["discriminator"] = model_dict["discriminator"].train()
    if gen_state: model_dict["generator"] = model_dict["generator"].train()

def pretrain_discriminator(model_dict, optimizer_dict, scheduler_dict,
                           dis_dataloader_params, vocab_size, positive_file,
                           negative_file, batch_size, epochs, use_cuda=False, temperature=1.0):
    discriminator = model_dict["discriminator"]
    d_optimizer = optimizer_dict["discriminator"]
    d_lr_scheduler = scheduler_dict["discriminator"]
    generate_samples(model_dict, negative_file, batch_size, use_cuda, temperature)
    dis_dataloader_params["positive_filepath"] = positive_file
    dis_dataloader_params["negative_filepath"] = negative_file
    dataloader = dis_data_loader(**dis_dataloader_params) # this is where data iterator is used
    cross_entropy = nn.CrossEntropyLoss() # this one is similar to NLL (negative log likelihood)
    if use_cuda: cross_entropy = cross_entropy.cuda()
    for epoch in range(epochs):
        for i, sample in enumerate(dataloader):
            d_optimizer.zero_grad()
            data, label = sample["data"], sample["label"] # initialize sample variables
            if use_cuda:
                data = data.cuda()
                label = label.cuda()
            outs = discriminator(data)
            loss = cross_entropy(outs["score"], label.view(-1)) + discriminator.l2_loss()
            loss.backward()
            d_optimizer.step()
            if i % LOG_MOD == 0:
                logging.debug("Pre-Discriminator loss: {:.5f}".format(loss))
        d_lr_scheduler.step()
    model_dict["discriminator"] = discriminator
    optimizer_dict["discriminator"] = d_optimizer
    scheduler_dict["discriminator"] = d_lr_scheduler
    return model_dict, optimizer_dict, scheduler_dict

#Adversarial training 
def adversarial_train(model_dict, optimizer_dict, scheduler_dict, dis_dataloader_params,
                      vocab_size, pos_file, neg_file, batch_size, gen_train_num = 4,
                      dis_train_num = 2, use_cuda = False, temperature = 1.0):
    """
        Get all the models, optimizer and schedulers
    """                     
    generator = model_dict["generator"]
    discriminator = model_dict ["discriminator"]
    worker = generator.worker
    manager = generator.manager

    m_optimizer = optimizer_dict["manager"]
    w_optimizer = optimizer_dict["worker"]
    d_optimizer = optimizer_dict["discriminator"]

    m_lr_scheduler = scheduler_dict["manager"]
    w_lr_scheduler = scheduler_dict["worker"]
    d_lr_scheduler = scheduler_dict["discriminator"]
    m_loss, w_loss = 0, 0 
    # Adversarial training for generator
    discriminator = discriminator.eval()
    for _ in range(gen_train_num):
        # Worker.
        w_optimizer.zero_grad()
        adv_rets = recurrent_func("adv")(model_dict, use_cuda)
        real_goal = adv_rets["real_goal"]
        prediction = adv_rets["prediction"]
        gen_token = adv_rets["gen_token"]
        rewards = get_rewards(model_dict, gen_token, use_cuda)
        w_loss = loss_func("adv_worker")(gen_token, prediction, rewards, vocab_size, use_cuda)
        w_loss.backward()
        w_optimizer.step()
        logging.debug("Adv-Manager loss: {:.5f} Adv-Worker loss: {:.5f}".format(m_loss, w_loss))
    
    m_lr_scheduler.step()
    w_lr_scheduler.step()
    
    del adv_rets
    del real_goal
    del prediction
    del gen_token
    del rewards
    
    # Adversarial training for discriminator
    discriminator = discriminator.train()
    generator = generator.eval()
    generate_samples(model_dict, neg_file, batch_size, use_cuda, temperature)
    for n in range(dis_train_num):
        dis_dataloader_params["positive_filepath"] = pos_file
        dis_dataloader_params["negative_filepath"] = neg_file
        dataloader = dis_data_loader(**dis_dataloader_params)
        cross_entropy = nn.CrossEntropyLoss()
        if use_cuda: cross_entropy = cross_entropy.cuda()
        for i, sample in enumerate(dataloader):
            d_optimizer.zero_grad()
            data, label = sample["data"], sample["label"]
            if use_cuda:
                data = data.cuda(non_blocking = True)
                label = label.cuda(non_blocking = True)
            outs = discriminator(data)
            loss = cross_entropy(outs["score"], label.view(-1)) + discriminator.l2_loss()
            loss.backward()
            d_optimizer.step()
        d_lr_scheduler.step()
        logging.debug("{}/{} Adv-Discriminator Loss: {:.5f}".format(n, dis_train_num, loss))
    generator = generator.train()
    # Save all changes
    model_dict["discriminator"] = discriminator
    generator.worker = worker
    generator.manager = manager
    model_dict["generator"] = generator
    optimizer_dict["manager"] = m_optimizer
    optimizer_dict["worker"] = w_optimizer
    optimizer_dict["discriminator"] = d_optimizer
    scheduler_dict["manager"] = m_lr_scheduler
    scheduler_dict["worker"] = w_lr_scheduler
    scheduler_dict["disciminator"] = d_lr_scheduler
    return model_dict, optimizer_dict, scheduler_dict

def get_model(prefix = "./", type = "latest") -> str:
    def get_no(s):
        standard = "checkpoint"
        for idx, ch in enumerate(s):
            if ch == "c":
                i = idx
                for ch_ref in standard:
                    if ch_ref == s[i]:
                        i += 1
                    else: break
                if i - idx == len(standard):
                    return int(s[i:])
        return -1
    rank = []
    for file in glob.glob(f"{prefix}saved_models/*.tar"):
        words = file.split('.')
        file_rank = get_no(words[-3])
        rank.append(file_rank)
    rank.sort()
    file_name = f"./saved_models/checkpoint{rank[0]}.pth.tar"
    if type == "latest":
        file_name = f"./saved_models/checkpoint{rank[-1]}.pth.tar"
    return file_name

def save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num, replace = False):
    file_name = "./saved_models/checkpoint" + str(ckpt_num) + ".pth.tar"
    torch.save({"model_dict": model_dict, "optimizer_dict": optimizer_dict, "scheduler_dict": scheduler_dict, "ckpt_num": ckpt_num}, file_name)
    if replace:
        oldest_ckpt = get_model(type = "oldest")
        os.remove(oldest_ckpt)

def restore_checkpoint(prefix = "./", ckpt_path = None):
    '''
    Restores model with latest id by default
    '''
    try:
        if ckpt_path == None:
            ckpt_path = get_model(prefix)
        combined_path = prefix + ckpt_path
        checkpoint = torch.load(combined_path, weights_only = False)
    except:
        logging.debug("[TM] No models are there to load.")
        return None
    return checkpoint

def train():
    global param_dict
    param_dict = get_arguments()
    use_cuda = torch.cuda.is_available()
    #Random seed
    torch.manual_seed(param_dict["train_params"]["seed"])
    #Pretrain step
    ckpt_num = 0
    checkpoint = restore_checkpoint()
    if checkpoint == None:
        model_dict = prepare_model_dict(use_cuda)
        lr_dict = param_dict["train_params"]["lr_dict"]
        optimizer_dict = prepare_optimizer_dict(model_dict, lr_dict)
        gamma = param_dict["train_params"]["decay_rate"]
        step_size = param_dict["train_params"]["decay_step_size"]
        scheduler_dict = prepare_scheduler_dict(optimizer_dict, gamma=gamma, step_size=step_size)
    else:
        model_dict = checkpoint["model_dict"]
        optimizer_dict = checkpoint["optimizer_dict"]
        scheduler_dict = checkpoint["scheduler_dict"]
        ckpt_num = checkpoint["ckpt_num"]
    #Pretrain discriminator
    logging.debug ("#########################################################################")
    discriminator = model_dict["discriminator"]
    logging.debug (f"Start Pretraining Discriminator... Model Size: {discriminator.get_model_wts()}")
    with open("./params/dis_data_params.json", 'r') as f:
        dis_data_params = json.load(f)
    if use_cuda:
        dis_data_params["pin_memory"] = True
    f.close()
    pos_file = dis_data_params["positive_filepath"]
    neg_file = dis_data_params["negative_filepath"]
    batch_size = param_dict["train_params"]["generated_num"]
    vocab_size = param_dict["leak_gan_params"]["discriminator_params"]["vocab_size"]
    epoch_pre_disc = 10 # Reduce time by keeping same Gen sample for 'epoch_pre_disc' 
    ## SET GENERATOR
    model_dict["generator"] = model_dict["generator"].eval()
    for i in range(0, param_dict["train_params"]["pre_dis_epoch_num"], epoch_pre_disc):
        logging.debug("Epoch: {}/{}  Pre-Discriminator".format(i, param_dict["train_params"]["pre_dis_epoch_num"]))
        model_dict, optimizer_dict, scheduler_dict = pretrain_discriminator(model_dict, optimizer_dict, scheduler_dict, dis_data_params, vocab_size = vocab_size, \
                                                                            positive_file = pos_file, negative_file = neg_file, batch_size = batch_size, \
                                                                            epochs = epoch_pre_disc, use_cuda = use_cuda)
    ## RESET GENERATOR
    model_dict["generator"] = model_dict["generator"].train()
    save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num)

    # Pretrain generator 
    logging.debug ("#########################################################################")
    generator = model_dict["generator"]
    logging.debug (f"Start Pretraining Generator... Model Size: {generator.get_model_wts()}")
    real_data_params = param_dict["real_data_params"]
    if use_cuda:
        real_data_params["pin_memory"] = True
    r_dataloader = real_data_loader(**real_data_params)
    ## SET DISCRIMINATOR
    model_dict["discriminator"] = model_dict["discriminator"].eval()
    for epoch in range(param_dict["train_params"]["pre_gen_epoch_num"]):
        logging.debug("Epoch: {}/{}  Pre-Generator".format(epoch, param_dict["train_params"]["pre_gen_epoch_num"]))
        model_dict, optimizer_dict, scheduler_dict = pretrain_generator(model_dict, optimizer_dict, scheduler_dict, r_dataloader, vocab_size=vocab_size, use_cuda=use_cuda)
    #Finish pretrain and save the checkpoint
    ## RESET DISCRIMINATOR
    model_dict["discriminator"] = model_dict["discriminator"].train()
    save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num)
    
    # Adversarial train of D and G
    logging.debug ("#########################################################################")
    logging.debug ("Start Adversarial Training...")
    vocab_size = param_dict["leak_gan_params"]["discriminator_params"]["vocab_size"]
    save_num = param_dict["train_params"]["save_num"] # Save checkpoint after this number of repetitions
    replace_num = param_dict["train_params"]["replace_num"]
    dis_train_num = 2 # Reduce time by keeping same generator sample for 'dis_train_num'
    for epoch in range(1, param_dict["train_params"]["total_epoch"], dis_train_num):
        logging.debug("Epoch: {}/{}  Adv".format(epoch, param_dict["train_params"]["total_epoch"]))
        model_dict, optimizer_dict, scheduler_dict = adversarial_train(model_dict, optimizer_dict, scheduler_dict, dis_data_params, vocab_size=vocab_size, pos_file=pos_file, neg_file=neg_file, \
                                                                       dis_train_num = dis_train_num, batch_size=batch_size, use_cuda=use_cuda)
        if (epoch-1)//dis_train_num % save_num == 0:
            ckpt_num += 1
            if ckpt_num % replace_num == 0:
                save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num, replace=True)
            else:
                save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num)

def eval(prefix = "./", n_samples = 4) -> float:
    bleu_score, bleu_score_batch = 0.0, 0.0
    pos_file = torch.from_numpy(np.load(prefix + "./formatted_data/positive_corpus.npy"))
    model_dict = restore_checkpoint(prefix)["model_dict"]
    use_cuda = torch.cuda.is_available()
    for _ in range(n_samples):
        gen_data = get_sample(model_dict, use_cuda = use_cuda, temperature = 1.0)
        bleu_score = Bleu.get_bleu(pos_file, gen_data) # gendata * pos_file bleu's score
        '''
            The sentence generated should match atleast one 
            positive sentence, hence taking max.
        '''
        bleu_score_batch += np.max(bleu_score, axis = 1).mean()
    return bleu_score_batch/n_samples