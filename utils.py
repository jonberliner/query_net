import torch
from torch.autograd import Variable
from torch.nn import functional as F

def apply_to_cartesian(fn):
    def wrapper(array1, array2=None, *args, **kwargs):
        array2 = array2 if array2 is not None else array1
        assert len(array1.shape) == 2
        assert len(array2.shape) == 2
        card1, dim1 = array1.shape 
        card2, dim2 = array2.shape
        cart = cartesian(array1, array2)

        out_flat = fn(cart, *args, **kwargs)

        shape = out_flat.shape[1:]
        return out_flat.view(card1, card2, *shape)
    return wrapper


def apply_to_image_coord_pairs(fn, return_coord_pairs=False):
    cart_fn = apply_to_cartesian(fn)

    def wrapper(images, *args, **kwargs):
        batch_size, dim, height, width = images.shape
        flat_images = images.permute([0, 2, 3, 1]).contiguous().view(batch_size, -1, dim)
        # height*width x dim
        output = torch.stack([cart_fn(fim, None, *args, **kwargs) 
                              for fim in flat_images])  # bs x (h*w) x (h*w) x outputsize

        if return_coord_pairs:
            coords = generate_grid_coords(height, width)
            output = (output, coords)
        return output
    return wrapper


def tile_leading_dim(matrix, n):
    """tile on leading dimension"""
    assert len(matrix.shape) == 2, 'currently only supports matrices'
    dim = matrix.shape[1]
    tiled = matrix.repeat(1, n).view(-1, dim)
    return tiled
    

def pairwise_l2_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is None:
        y = x
        y_norm = x_norm
    else:
        y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, min=0.0)


# extended from https://discuss.pytorch.org/t/generating-a-meshgrid-or-itertools-product-in-pytorch/1973
def cartesian(array1, array2=None, cycle_array1_slower=True, cat=False):
    array2 = array2 if array2 is not None else array1
    assert len(array1.shape) == 2
    assert len(array2.shape) == 2
    card1, dim = array1.shape
    card2, dim2 = array2.shape

    if cycle_array1_slower:
        rep1 = tile_leading_dim(array1, card2)
        rep2 = array2.repeat(card1, 1)
    else:
        rep1 = array1.repeat(card2, 1)
        rep2 = tile_leading_dim(array2, card1)

    # concat into single array?
    cart = torch.cat([rep1, rep2], -1) if cat else (rep1, rep2)
    return cart


def mapn_on_rows(function, *arrs):
    """passing n arrays with same leading dimension and apply function to each
    row-wise tuple"""
    return torch.stack([function(*arrs_i) for arrs_i in zip(*arrs)])


def generate_grid_coords(h, w=None, hbounds=None, wbounds=None):
    w = w or h
    hbounds = hbounds or (0, h-1)
    wbounds = wbounds or (0, w-1)

    x = torch.linspace(hbounds[0], hbounds[1], h).unsqueeze(1)
    y = torch.linspace(wbounds[0], wbounds[1], w).unsqueeze(1)

    return cartesian(x, y, cat=True)


@apply_to_cartesian
def pairwise_dists(cart, p=2):
    return torch.norm(cart[0] - cart[1], p=p, dim=1)


@apply_to_cartesian
def pairwise_mul(cart):
    return cart[0] * cart[1]


@apply_to_cartesian
def pairwise_se(cart):
    return (cart[0] - cart[1]).pow(2.).mul(-0.5).exp()


@apply_to_image_coord_pairs
def image_pairwise_mul(cart):
    return cart[0] * cart[1]


@apply_to_image_coord_pairs
def image_pairwise_add(cart):
    return cart[0] + cart[1]


SMALL = 1e-6
def soft_normalized_cut_loss(logits, pixel_pair_dists=None):
    log_probs = F.log_softmax(logits, dim=1)
    pixel_pair_cat_agreement = image_pairwise_add(log_probs)
    # prep for l2 distances
    pixel_pair_cat_agreement = pixel_pair_cat_agreement.permute([0, 3, 1, 2])
        # pixel_pair_dists = Variable(pairwise_dists(coords))
    if pixel_pair_dists is None:
        coords = generate_grid_coords(log_probs.shape[2], log_probs.shape[3])
        pixel_pair_dists = Variable(pairwise_l2_distances(coords))
    distance_weighted_agreement = pixel_pair_cat_agreement \
                                  * (1. / (pixel_pair_dists + SMALL))
    return -distance_weighted_agreement.sum()


def soft_normalized_cut_loss2(logits, pixel_pair_dists):
    probs = F.softmax(logits, dim=1)
    batch_size, ch, height, width = logits.shape
    probs_flat = probs.view(batch_size, ch, -1)

    # numerator
    nleft = probs_flat
    nright = (probs_flat.unsqueeze(-1) * pixel_pair_dists).sum(-1)
    # denom
    dleft = probs_flat
    dright = pixel_pair_dists.sum(1)

    # per-class
    numers = (nleft * nright).sum(-1)
    denoms = (dleft * dright).sum(-1)

    # agg over classes
    loss = ch - (numers * (1. / denoms)).sum(1)

    return loss.sum()


def soft_normalized_cut_loss_local(logits, pixel_pair_dists=None, window_size=7,
                                   length_scale=1., signal_sd=1.,
                                   maybe_log=False):
    """local apprx to soft norm cut using a conv kernel"""

    batch_size, ch, height, width = logits.shape
    padding = window_size // 2

    coord_grid = generate_grid_coords(window_size, window_size) - padding
    se_kernel = torch.norm(coord_grid, p=2, dim=1)\
                .mul(-(1. / (2. * (length_scale**2))))\
                .exp()\
                .mul(signal_sd**2)
    se_kernel = Variable(se_kernel)

    probs = F.softmax(logits, dim=1)

    padded_probs = F.pad(probs, tuple([padding]*4), value=1./ch)
    expanded_sek = se_kernel.view(1, 1, window_size, window_size)\
                            .expand(ch, ch, -1, -1)
    # in log space
    if maybe_log:
        log_probs = F.log_softmax(logits, dim=1)
        numer = F.conv2d(padded_probs, expanded_sek).log() + log_probs
        denom = se_kernel.sum().log() + log_probs
        losseses = numer.view(batch_size, ch, -1).sum(-1)\
                - denom.view(batch_size, ch, -1).sum(-1)

    else:
        numer = F.conv2d(padded_probs, expanded_sek) * probs
        denom = se_kernel.sum() * probs
        losseses = numer.view(batch_size, ch, -1).sum(-1)\
                / denom.view(batch_size, ch, -1).sum(-1)

    losses = ch - losseses.sum(-1)
    loss = losses.sum()

    return loss
