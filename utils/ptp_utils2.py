import abc

import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List
from torchvision import transforms as T

from diffusers.models.attention import Attention

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img


class AttendExciteCrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention,hidden_states: torch.FloatTensor,encoder_hidden_states = None,
                    attention_mask = None,temb = None,scale: float = 1.0,) -> torch.Tensor:
        residual = hidden_states
        is_cross = encoder_hidden_states is not None

        args = (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def register_attention_control(model, controller):

    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteCrossAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

def cal_threshold(img):
    """
    img: 1*h*w
    """
    img = img.detach().cpu().numpy().transpose(1,2,0)
    img = (img * 255).astype(np.uint8)
    # 先进行高斯滤波，再使用Otsu阈值法
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    blur = blur.astype("uint8")
    ret3, th3 = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def get_crossMask(attention_store: AttentionStore,
                    res: int,
                    from_where: List[str],
                    select: int,
                    indices: List[int],
                    save_name: str) -> torch.Tensor:
    """get cross-attn mask"""

    toImg = T.ToPILImage()
    corss_attn_map = aggregate_attention(attention_store, res, from_where, True, select) #h w 77

    h, w, seq_len = corss_attn_map.shape
    corss_attn_map = corss_attn_map.permute(2,0,1) #77 h w
    amap_glo = np.zeros((h,w))

    for index in indices:
        attn_map = corss_attn_map[index]
        attn_map = (attn_map - attn_map.min())/(attn_map.max() - attn_map.min())

        attn_map = attn_map**2
        attn_map = (attn_map - attn_map.min())/(attn_map.max() - attn_map.min())

        attn_map = attn_map**2
        attn_map = (attn_map - attn_map.min())/(attn_map.max() - attn_map.min())


        amap = attn_map.reshape(1,32,32).float()
        # amap = amap/amap.sum()
        amap = cal_threshold(amap)
        amap_glo = amap_glo + amap.reshape(32,32)

    mask = amap_glo
    mask = torch.from_numpy(mask)
    toImg(mask.reshape(1,h,w)).save(save_name)
    mask = mask != 0 #bool tensor
    attention_store.mask = mask    



def attn_refine(amap, coor_index = None):
    """
    amap: batch_size*head_nums h*w h*w
    coor_index: (h*w), bool tensor
    """
    bh, tnum, res = amap.shape
    grid_size = h = w = int(torch.sqrt(torch.tensor(res)))
    
    amap = amap[:,coor_index,:] # bh tnum res
    bh, tnum, res = amap.shape

    x = torch.arange(grid_size).float()
    y = torch.arange(grid_size).float()
    grid_x, grid_y = torch.meshgrid(x, y)
    grid_xy = torch.stack((grid_x, grid_y), dim=2).cuda()  # shape: h*w*2
    amap = amap.reshape(bh, tnum, h, w)

    # mu = torch.einsum('bijk,jkl->bil', amap, grid_xy) #shape: bh tnum 2
    # mu = mu.reshape(bh,tnum,1,1,2)

    mu = grid_xy.reshape(-1,2)[coor_index, :].reshape(1,tnum, 1,1,2)
    # mu = torch.tensor([12.,20.]).reshape(1,1,1,1,1,2)
    mu = mu.repeat(bh,1,1,1,1) #bh tmnm 1 1 2
    xy_norm = grid_xy.view(1, 1, h, w, 2) - mu #shape: bh tnum h w 2

    xy_norm = xy_norm.reshape(-1,2,1)
    xy_square = torch.bmm(xy_norm, xy_norm.permute(0,2,1)).reshape(bh, tnum, h, w, 2, 2) #bh, tnum, h, w,2,2
    
    sigma = torch.einsum('bijk,bijklm->bilm',amap, xy_square) #bh tnum 2 2
    # sigma[:,:] = torch.tensor([[1,0.],[0,1.]]).cuda()

    inv_sigma = torch.linalg.inv(sigma) #bh tnum 2 2
    inv_sigma = inv_sigma.reshape(bh, tnum, 1, 1, 2, 2)
    inv_sigma = inv_sigma.repeat(1, 1, h, w, 1, 1) #bh tnum h w 2 2
    
    dis = torch.bmm(xy_norm.permute(0,2,1), inv_sigma.reshape(-1,2,2)) #-1 1 2
    dis = torch.bmm(dis, xy_norm).reshape(bh,tnum,h,w) #bh, tnum, h, w

    dis = torch.sqrt(dis) #bh tnum h w

    dis2 = (dis - dis.amin(dim = (-2,-1), keepdim = True)) / (dis.amax(dim = (-2,-1), keepdim = True) - dis.amin(dim = (-2,-1), keepdim = True))
    dis2 = torch.exp(-dis2/0.5) # bh tnum h w
    
    amap = amap * dis2
    amap = (amap - amap.amin(dim = (-2,-1), keepdim = True))/(amap.amax(dim = (-2,-1), keepdim = True) - amap.amin(dim = (-2,-1), keepdim = True))
    # amap = amap**1.5

    amap = amap/amap.sum(dim = (-2,-1), keepdim=True)
    
    return amap.reshape(bh,tnum,h*w)


class CompactAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention,hidden_states: torch.FloatTensor,encoder_hidden_states = None,
                    attention_mask = None,temb = None,scale: float = 1.0,) -> torch.Tensor:
        residual = hidden_states
        is_cross = encoder_hidden_states is not None

        args = (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        if attn.eot is not None:
            value2 = attn.to_v(attn.eot, *args)
            # print(value2.shape, value.shape)
            value[1][8:] = value2[0][8:]

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        

        #---------------------------------
        # if attn.time is not None and attn.time > 15:
        #     mask = self.attnstore.mask
        #     mask = mask.reshape(-1)
        #     mask = ~mask
        #     bh, h, w = attention_probs.shape
        #     attention_probs[bh//2:, mask, :] = attn_refine(attention_probs[bh//2:], mask)
        #---------------------------------


        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def register_self_time(pipe, i):
    for name, module in pipe.unet.named_modules():
        # if name in attn_greenlist:
        if (name.startswith("mid_block")) and name.endswith("attn1"):
        # if name.endswith("attn1"):
        # if name.startswith("down_blocks.2") and name.endswith("attn1"):
            setattr(module, 'time', i)

def register_self_eot(pipe, eot):
    for name, module in pipe.unet.named_modules():
        setattr(module, 'eot', eot)
