import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from utils import generate_grid_coords, tile_leading_dim

from numpy.random import uniform

from pyt.modules import MLP
from deepset2d import DeepSet2d

def categorical_entropy(logits):
    log_probs = F.log_softmax(logits)
    probs = F.softmax(logits)
    return -(log_probs * probs).sum(-1)

TASK = 'classify'
DATASET = 'mnist'
DATA_DIR = '/Users/jsb/datasets'

import pyt.testing 
from torchvision.transforms import Compose, Resize, ToTensor
transform = Compose([Resize(32), ToTensor()])
example_im = next(iter(pyt.testing.quick_dataset(DATASET, 
                                                DATA_DIR,
                                                transform=transform, 
                                                batch_size=1)['train']))[0]

dim_input = example_im.shape[1]
shape_input = example_im.shape[2:]
dim_loc = len(shape_input)
n_class = 10

dim_em_obs = 59
dim_em_loc = 57
dim_em_obsloc = 129

obs_encoder = MLP(dim_input, dim_em_obs, [128], act_fn=F.elu)#, norm='batch')
loc_encoder = MLP(dim_loc, dim_em_loc, [128], act_fn=F.elu)#, norm='batch')
obsloc_encoder = MLP(dim_em_obs + dim_em_loc, dim_em_obsloc, [128], act_fn=F.elu)#, norm='batch')
classifier = MLP(dim_em_obsloc, n_class, [128], act_fn=F.elu)#, norm='batch')

net = DeepSet2d(obs_encoder, 
                loc_encoder, 
                obsloc_encoder, 
                classifier, 
                subsample=None)
                # subsample=0.4)

# # control model
# net = MLP(1024, 10, [32, 32, 32], norm='batch', p_drop=0.2, act_fn=F.elu)
# transform = Compose([Resize(32), ToTensor(), lambda im: im.view(-1)])

pyt.testing.test_over_dataset(model=net,
                            dataset=DATASET, 
                            data_dir=DATA_DIR,
                            task=TASK, 
                            transform=transform,
                            p_splits={'train': 0.02, 'val': 0.01},
                            training_kwargs={'batch_size': 32, 'n_epoch': 100},
                            balanced=True,
                            include_test_data_loader=False)

# # inference_net = MLP(1024, 10, [128], norm='batch', act_fn=Swish())

# class QueryNet(nn.Module):
#     def __init__(self, 
#                  inference_net, 
#                  next_net, 
#                  entropy_net,
#                  stop_net,
#                  dim_input, 
#                  dim_output):
#         super().__init__()
#         self.inference_net = inference_net
#         self.next_net = next_net
#         self.entropy_net = entropy_net
#         self.stop_net = stop_net

#     def forward(self, inputs, n_next=10):
#         max_queries = inputs.shape[1] // n_next

#         batch_size, in_channels = inputs.shape
#         query_mask = Variable(torch.zeros(batch_size, in_channels))

#         # initialize output
#         output_logits = Variable(torch.zeros(batch_size, self.dim_output))
#         prev_ent = categorical_entropy(output_logits)

#         # initialize masked input
#         _inputs = query_mask * inputs

#         # request next observation
#         pnext = self.next_net(_inputs, outputs)
#         _, inext = pnext.topk(n_next, )

#         stop_signal = self.stop_net(Variable(torch.cat([_inputs, output_logits], 1)))

#         # unmask next input
#         query_mask = query_mask.scatter(1, inext, 1.)

#         # train entropy net to predict entropy reduction upon revealing next observation
#         next_onehots = Variable(torch.zeros_like(_inputs).scatter(1, inext, 1.))
#         entropy_net_inputs = torch.cat([_inputs, next_onehots, output_logits], 1)
#         hat_delta_entropy = self.entropy_net(entropy_net_inputs)

#         # observe next observation
#         output_logits = self.inference_net(_inputs)

#         # get drop in entropy
#         curr_ent = categorical_entropy(prev_ent)
#         delta_entropy = curr_ent - prev_ent

#         # loss for entropy predictor
#         ent_loss = mse(hat_delta_entropy, delta_entropy)

#         # loss for categorizer

#         # best_secondbest = output_logits.topk(2, dim=1)
#         # reduction_bvsb = best_secondbest

#         # 

#         _inputs, next_onehots



#     # def forward(self, inputs):
#     #     obss, locs = self.to_set_representation(inputs)

#     #     hidden_states = torch.zeros(batch_size, dim_output)

#     #     einputs = cartesian(hidden_states, locs, cat=True)

#     #     rep_hidden_states = tile_leading_dim(hidden_states, locs.shape[0])
#     #     einputs = torch.cat(rep_hidden_states)
#     #     expected_entropy_decrease = self.entropy_net()
