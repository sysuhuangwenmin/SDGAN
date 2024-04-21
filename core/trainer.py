"""
"""
from networks import Gen
from utils import weights_init, get_model_list
import torch
import torch.nn as nn
import os
import copy



def update_average(model_tgt, model_src, beta=0.99):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

class SDGAN(nn.Module):
    def __init__(self, hyperparameters):
        super(SDGAN, self).__init__()
        self.gen = Gen(hyperparameters)


        self.noise_dim = hyperparameters['noise_dim']  #32
        self.hyperparameters = hyperparameters

    def forward(self, args, mode):
        if mode == 'gen':
            f=self.gen_losses(*args)
            return f
        elif mode == 'dis':
            return self.dis_losses(*args)
        else:
            pass

class SDGAN_Trainer(nn.Module):
    def __init__(self, hyperparameters, multi_gpus=False):
        super(SDGAN_Trainer, self).__init__()
        # Initiate the networks
        self.multi_gpus = multi_gpus
        self.models = SDGAN(hyperparameters)
        self.models.gen_test = copy.deepcopy(self.models.gen)


    def resume(self, checkpoint_dir, hyperparameters):  #
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.models.gen.load_state_dict(state_dict['gen'])
        self.models.gen_test.load_state_dict(state_dict['gen_test'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.models.dis.load_state_dict(state_dict['dis'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        for state in self.dis_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
        for state in self.gen_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        print('Resume from iteration %d' % iterations)
        return iterations
    

    def save(self, snapshot_dir, iterations):
        this_model = self.models.module if self.multi_gpus else self.models
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'gen': this_model.gen.state_dict(), 'gen_test': this_model.gen_test.state_dict()}, gen_name)
        torch.save({'dis': this_model.dis.state_dict()}, dis_name)
        torch.save({'dis': self.dis_opt.state_dict(),
                    'gen': self.gen_opt.state_dict()}, opt_name)
