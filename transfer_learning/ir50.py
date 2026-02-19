from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb


##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


# i = 0

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))
        i = 0

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        # print(shortcut.shape)
        # print('---s---')
        res = self.res_layer(x)
        # print(res.shape)
        # print('---r---')
        # i = i + 50
        # print(i)
        # print('50')
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    # print('50')


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks


class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir', num_classes=6):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        
        # Define bodies using the blocks list
        # blocks[0] -> body1 (64 channels)
        # blocks[1] -> body2 (128 channels)
        # blocks[2] -> body3 (256 channels)
        # blocks[3] -> body4 (512 channels)
        
        self.body1 = Sequential(*[unit_module(b.in_channel, b.depth, b.stride) for b in blocks[0]])
        self.body2 = Sequential(*[unit_module(b.in_channel, b.depth, b.stride) for b in blocks[1]])
        self.body3 = Sequential(*[unit_module(b.in_channel, b.depth, b.stride) for b in blocks[2]])
        self.body4 = Sequential(*[unit_module(b.in_channel, b.depth, b.stride) for b in blocks[3]])

        # Classification Head for Emotion (num_classes)
        # Assuming input size of 112x112 -> 7x7 at the end of body4
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512),
                                       ReLU(inplace=True), 
                                       Linear(512, num_classes)) 
                                       

    def forward(self, x):
        x = F.interpolate(x, size=112, mode='bilinear', align_corners=True)
        x = self.input_layer(x)
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x = self.body4(x)
        
        x = self.output_layer(x)
        return x

def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers = []
    
    # Define block ranges for IR-50
    # body1: 0-2 (3 blocks)
    # body2: 3-6 (4 blocks)
    # body3: 7-20 (14 blocks)
    # body4: 21-23 (3 blocks)
    
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        
        original_k = k
        
        # Handle body key remapping
        if k.startswith('body.'):
            # Format is body.Index.Submodule...
            parts = k.split('.')
            if parts[1].isdigit():
                idx = int(parts[1])
                rest = '.'.join(parts[2:])
                
                if 0 <= idx < 3:
                    new_k = f"body1.{idx}.{rest}"
                elif 3 <= idx < 7:
                    new_k = f"body2.{idx - 3}.{rest}"
                elif 7 <= idx < 21:
                    new_k = f"body3.{idx - 7}.{rest}"
                elif 21 <= idx < 24:
                    new_k = f"body4.{idx - 21}.{rest}"
                else:
                    new_k = k # Should not happen for IR-50
                
                k = new_k

        if k in model_dict:
            if model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                print(f"Size mismatch: {k} (ckpt: {v.size()}, model: {model_dict[k].size()})")
        else:
            pass # key not in model
            
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print(f'Weights loaded: {len(matched_layers)} tensors matched')
    return model
