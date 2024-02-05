# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.distributed as dist


def autopad(k, p=None, d=1):  
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):  
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = SiLU() 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))    

class C2f(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e) 
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    # SPP结构，5、9、13最大池化核的最大池化。
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

def get_gpu_states(fwd_gpu_devices):
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_states

def get_gpu_device(*args):

    fwd_gpu_devices = list(set(arg.get_device() for arg in args
                               if isinstance(arg, torch.Tensor) and arg.is_cuda))
    return fwd_gpu_devices

def set_device_states(fwd_cpu_state, devices, states) -> None:
    torch.set_rng_state(fwd_cpu_state)
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)

def detach_and_grad(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = True
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)

def get_cpu_and_gpu_states(gpu_devices):
    return torch.get_rng_state(), get_gpu_states(gpu_devices)

class ReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions  = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0  = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0*alpha0
            ctx.cpu_states_1, ctx.gpu_states_1  = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1*alpha1
            ctx.cpu_states_2, ctx.gpu_states_2  = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2*alpha2
            ctx.cpu_states_3, ctx.gpu_states_3  = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3*alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1 ,c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
            torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
            torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
            torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            
            g3_up = g3_right
            g3_left = g3_up*alpha3 ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)                    
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1/alpha3)*(c3 - oup3) ## feature reverse
            g2_up = g2_right+ c2.grad
            g2_left = g2_up*alpha2 ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)          
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left*alpha3 ##alpha3 update
            torch.autograd.backward(cout3, g3_up)
            
            with torch.no_grad():
                c2_left = (1/alpha2)*(c2 - oup2) ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right+c1.grad
            g1_left = g1_up*alpha1 ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)     
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left*alpha2 ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1/alpha1)*(c1 - oup1) ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up*alpha0 ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left ## Fusion
            
            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)     
            oup0 = l0(x, c1_left)            
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left*alpha1 ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1/alpha0)*(c0 - oup0) ## feature reverse
            gx_up = x.grad ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left*alpha0 ##alpha0 update
            torch.autograd.backward(cout0, g0_up)
        
        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left


class Fusion(nn.Module):
    def __init__(self, level, channels, first_col) -> None:
        super().__init__()
        
        self.level = level
        self.first_col = first_col
        self.down = Conv(channels[level-1], channels[level], 3, 2) if level in [1, 2, 3] else nn.Identity()
        if not first_col:
            self.up = nn.Sequential(Conv(channels[level+1], channels[level]), nn.Upsample(scale_factor=2, mode='nearest')) if level in [0, 1, 2] else nn.Identity()            

    def forward(self, *args):

        c_down, c_up = args
        
        if self.first_col:
            x = self.down(c_down)
            return x
        
        if self.level == 3:
            x = self.down(c_down)
        else:
            x = self.up(c_up) + self.down(c_down)
        return x

class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel, first_col, last_col=False) -> None:
        super().__init__()
        self.fusion = Fusion(level, channels, first_col)
        if last_col:
            modules = [eval(f'{kernel}')(channels[level], channels[level], layers[level], True), SPPF(channels[level], channels[level], k=5)]
        else:
            modules = [eval(f'{kernel}')(channels[level], channels[level], layers[level], True)]
        self.blocks = nn.Sequential(*modules)
    def forward(self, *args):
        x = self.fusion(*args)
        x = self.blocks(x)
        return x

class RevcolNet(nn.Module):
    def __init__(self, channels, layers, kernel, first_col, last_col, save_memory) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 

        # C2F
        self.level0 = Level(0, channels, layers, kernel, first_col)
        self.level1 = Level(1, channels, layers, kernel, first_col)
        self.level2 = Level(2, channels, layers, kernel, first_col)
        self.level3 = Level(3, channels, layers, kernel, first_col, last_col)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args

        c0 = (self.alpha0)*c0 + self.level0(x, c1)
        c1 = (self.alpha1)*c1 + self.level1(c0, c2)
        c2 = (self.alpha2)*c2 + self.level2(c1, c3)
        c3 = (self.alpha3)*c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):
        
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign

class RevCol(nn.Module):
    def __init__(self, kernel='C2f', base_channels=32, base_depth=1, num_subnet=5, save_memory=True) -> None:
        super().__init__()
        self.num_subnet = num_subnet
        self.channels = [base_channels*2, base_channels*4, base_channels*8, base_channels*16]
        self.layers = [base_depth, base_depth*2, base_depth*2, base_depth]
        
        self.stem = nn.Sequential(
            Conv(3, self.channels[0] // 2, 3, 2),
            Conv(self.channels[0] // 2, self.channels[0], 3, 2)
        )

        for i in range(num_subnet):
            first_col = True if i == 0 else False
            last_col = True if i == num_subnet-1 else False
            self.add_module(f'revcolnet{str(i)}', RevcolNet(self.channels, self.layers, kernel, first_col, last_col, save_memory=save_memory))

    def forward(self, x):
        c0, c1, c2, c3 = 0, 0, 0, 0
        x = self.stem(x)        
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'revcolnet{str(i)}')(x, c0, c1, c2, c3)       

        return [c1, c2, c3]

