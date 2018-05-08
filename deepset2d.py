import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils import generate_grid_coords, tile_leading_dim
from numpy.random import uniform

class DeepSet2d(nn.Module):
    def __init__(self, 
                 obs_encoder,
                 loc_encoder,
                 obsloc_encoder,
                 classifier,
                 subsample=None,
                 non_negative=False):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.loc_encoder = loc_encoder
        self.obsloc_encoder = obsloc_encoder
        self.classifier = classifier
        self.subsample = subsample
        self.non_negative = non_negative


    def to_em_set(self, obss, locs, return_indiv=False):
        batch_size, n_obs, dim_obs = obss.shape
        # location embeddings
        if len(locs.shape) == 2:
            # save batch_size-1 passes through loc_encoder
            em_loc = self.loc_encoder(locs).repeat(batch_size, 1)
        else:
            em_loc = self.loc_encoder(locs.view(batch_size * n_obs, -1))

        # observation embeddings
        em_obss = self.obs_encoder(obss.view(batch_size * n_obs, -1))
        
        # obss conditioned on locs
        em_obsloc = self.obsloc_encoder(torch.cat([em_obss, em_locs], -1))\
                    .view(batch_size, n_obs, -1)
        if self.non_negative:
            em_obsloc = F.softplus(em_obsloc)

        # integrate set evidence
        em_set = em_obsloc.sum(1)
        if return_indiv:
            return em_set, em_obsloc
        else:
            return em_set


    def subsample_sets(self, obss, locs, subsample=None, replace=True):
        if subsample:
            if type(subsample) is tuple:
                subsample = uniform(subsample[0], subsample[1])
            assert subsample > 0. and subsample <= 1.
            if subsample < 1.:
                batch_size, n_all_loc, dim_obs = obss.shape
                n_obs = max(int(n_all_loc * subsample), 1)
                # different subset for every image
                # fast, but with possibility of replacement
                if replace:
                    i_loc = Variable(torch.LongTensor(batch_size * n_obs)
                                    .random_(n_all_loc))
                # slower, but without replacement
                else:
                    i_loc = Variable(torch.cat([torch.randperm(n_all_loc)[:n_obs]
                                                for _ in range(batch_size)], 0))
                obss = obss.view(-1, dim_obs)\
                           .index_select(0, i_loc)\
                           .view(batch_size, n_obs, -1)
                locs = locs.index_select(0, i_loc).view(batch_size, n_obs, -1)
        else:
            locs = locs.unsqueeze(0).expand(batch_size, -1, -1)
        return obss, locs


    def to_set_representation(self, inputs):
        batch_size, in_channels, height, width = inputs.shape
        observations = inputs.view(batch_size, in_channels, -1).permute(0, 2, 1).contiguous()  # bs x ncoord x ch
        locations = Variable(generate_grid_coords(height, width, (-10, 10), (-10, 10)))
        return observations, locations


    def forward(self, images, subsample=None, replace=True):
        if self.training:
            subsample = subsample or self.subsample
        obss, locs = self.to_set_representation(images)
        obss, locs = self.subsample_sets(obss, locs, subsample, replace)
        em_set = self.to_em_set(obss, locs)  # bs x dim_hid
        logits = self.classifier(em_set)
        return logits


if __name__ == '__main__':
    import pyt.testing 
    from pyt.modules import MLP
    from torchvision.transforms import Compose, Resize, ToTensor
    from os.path import expanduser

    TASK = 'classify'
    DATASET = 'mnist'
    home_dir = expanduser("~")
    DATA_DIR = os.path.join(home_dir, 'datasets')

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

    pyt.testing.test_over_dataset(model=net,
                                dataset=DATASET, 
                                data_dir=DATA_DIR,
                                task=TASK, 
                                transform=transform,
                                p_splits={'train': 0.02, 'val': 0.01},
                                training_kwargs={'batch_size': 32, 'n_epoch': 100},
                                balanced=True,
                                include_test_data_loader=False)
