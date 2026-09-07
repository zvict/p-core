import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch import autocast
import torch.nn.functional as F
import tinycudann as tcnn
import warnings
import typing as T
from .utils import activation_func


def get_mapping_mlp(args, use_amp=False, amp_dtype=torch.float16):
    return MappingMLP(args.mapping_mlp, inp_dim=args.shading_code_dim, out_dim=args.mapping_mlp.out_dim, use_amp=use_amp, amp_dtype=amp_dtype)


class MLP(nn.Module):
    def __init__(self, inp_dim=2, num_layers=3, num_channels=128, out_dim=2, act_type="relu", last_act_type="none",
                 use_wn=False, a=1., b=1., trainable=False, skip_layers=[], bias=True, half_layers=[], residual_layers=[],
                 residual_dims=[]):
        super(MLP, self).__init__()
        self.skip_layers = skip_layers
        self.residual_layers = residual_layers
        self.residual_dims = residual_dims
        assert len(residual_dims) == len(residual_layers)
        wn = weight_norm if use_wn else lambda x, **kwargs: x
        layers = [nn.Identity()]
        num_layers = num_layers + 1
        for i in range(num_layers):
            cur_inp = inp_dim if i == 0 else num_channels
            cur_out = out_dim if i == num_layers - 1 else num_channels
            if (i+1) in half_layers:
                cur_out = cur_out // 2
            if i in half_layers:
                cur_inp = cur_inp // 2
            if i in self.skip_layers:
                cur_inp += inp_dim
            if i in self.residual_layers:
                cur_inp += self.residual_dims[residual_layers.index(i)]
            layers.append(
                wn(nn.Linear(cur_inp, cur_out, bias=bias), name='weight'))
            layers.append(activation_func(act_type=act_type,
                          num_channels=cur_out, a=a, b=b, trainable=trainable))
        layers[-1] = activation_func(act_type=last_act_type,
                                     num_channels=out_dim, a=a, b=b, trainable=trainable)
        assert len(layers) == 2 * num_layers + 1
        self.model = nn.ModuleList(layers)

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, residuals=[]):
        skip_layers = [i*2+1 for i in self.skip_layers]
        residual_layers = [i*2+1 for i in self.residual_layers]
        assert len(residuals) == len(self.residual_layers)
        # print(skip_layers)
        inp = x
        for i, layer in enumerate(self.model):
            if i in skip_layers:
                x = torch.cat([x, inp], dim=-1)
            if i in residual_layers:
                x = torch.cat([x, residuals[residual_layers.index(i)]], dim=-1)
            x = layer(x)
        return x


class MappingMLP(nn.Module):
    def __init__(self, args, inp_dim=2, out_dim=2, use_amp=False, amp_dtype=torch.float16):
        super(MappingMLP, self).__init__()
        self.args = args
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.model = MLP(inp_dim=inp_dim, num_layers=args.num_layers, num_channels=args.dim, out_dim=out_dim,
                         act_type=args.act, last_act_type=args.last_act, use_wn=args.use_wn)
        print("Mapping MLP:\n", self.model)

    def forward(self, x):

        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            out = self.model(x)
            return out


class TopkMLP(nn.Module):
    def __init__(self, args, num_feats=3, device='cuda', use_amp=False, amp_dtype=torch.float16):
        super(TopkMLP, self).__init__()
        self.args = args
        self.loss_type = args.loss_type.lower()
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.means = args.means
        self.bias = None
        
        if args.learn_bias:
            self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device), requires_grad=True)
        
        if args.init_weights == []:
            assert args.learn == True
            if args.type == 1:    # including d2r
                self.weights = nn.Parameter(torch.cat([torch.ones(1, device=device), torch.zeros(num_feats - 1, device=device)], dim=0), requires_grad=True)
            elif args.type == 2:    # excluding d2r
                self.weights = nn.Parameter(torch.zeros(num_feats - 1, dtype=torch.float32, device=device), requires_grad=True)
        else:
            assert len(args.init_weights) == num_feats - 1
            if args.type == 1:
                self.weights = nn.Parameter(torch.cat([torch.ones(1, device=device), torch.tensor(args.init_weights, device=device, dtype=torch.float32)], dim=0), requires_grad=args.learn)
            elif args.type == 2:
                self.weights = nn.Parameter(torch.tensor(args.init_weights, device=device, dtype=torch.float32), requires_grad=args.learn)

        self.weight_act = activation_func(act_type=args.weight_act)

    @torch.no_grad()
    def get_features(self, feature_list, x=None):
        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            if self.means == []:
                self.means = [f.mean().item() for f in feature_list]
            elif self.args.means == []:
                cur_means = [f.mean().item() for f in feature_list]
                self.means = [(m + cm) / 2 for m, cm in zip(self.means, cur_means)]

            if self.args.minus_mean:
                feature_list = [f - m for f, m in zip(feature_list, self.means)]
            
            weights = self.weights.clone()
            weights[self.args.act_idxs] = self.weight_act(weights[self.args.act_idxs])

            if self.args.type == 1:   # including D2R
                features = torch.stack(feature_list, dim=-1)
                scores = torch.sum(features * weights, dim=-1)
            elif self.args.type == 2:   # excluding D2R
                features = torch.stack(feature_list[1:], dim=-1)
                scores = torch.sum(features * weights, dim=-1) + feature_list[0]
            else:
                raise NotImplementedError(f"Invalid type: {self.args.type}")

            return scores
        
    def forward(self, feature_list, attn_scores, x=None, step=-1):
        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            with torch.autograd.grad_mode.set_grad_enabled(self.args.learn):
                if self.args.minus_mean:
                    feature_list = [f - m for f, m in zip(feature_list, self.means)]

                weights = self.weights.clone()
                weights[self.args.act_idxs] = self.weight_act(weights[self.args.act_idxs])

                if self.args.type == 1:
                    features = torch.stack(feature_list, dim=-1)
                    scores = torch.sum(features * weights, dim=-1)
                elif self.args.type == 2:
                    features = torch.stack(feature_list[1:], dim=-1)
                    scores = torch.sum(features * weights, dim=-1) + feature_list[0]
                else:
                    raise NotImplementedError(f"Invalid type: {self.args.type}")
                
                if step % 200 == 0:
                    print("topk mlp weights: ", self.weights.tolist(), weights.tolist())
                    if self.means is not None:
                        print("topk mlp means: ", self.means)
                    if self.bias is not None:
                        print("topk mlp bias: ", self.bias.item())

                if self.bias is not None:
                    scores += self.bias

                if self.loss_type == "bce":
                    labels = torch.ones_like(attn_scores, requires_grad=False)
                    medians = torch.quantile(attn_scores.float(), q=self.args.quantile, dim=-1, keepdim=True)[0]
                    labels[attn_scores > medians] = 0   # Highers attn scores should have lower score values, since topk selects the lowest values
                    if self.args.quantile != 0.5 and self.args.weight_quantile:
                        # weight the loss
                        weights = torch.ones_like(attn_scores, requires_grad=False)
                        weights[attn_scores > medians] = 0.5 / (1 - self.args.quantile)
                        weights[attn_scores <= medians] = 0.5 / self.args.quantile
                    else:
                        weights = None
                    loss = nn.BCEWithLogitsLoss(weight=weights)(scores * self.args.logit_scale, labels)

                return loss


_GAIN_ALIASES = {"silu": "relu", "swish": "relu"}

_INIT_METHODS = {
    "normal": lambda w, gain, a, mode: nn.init.normal_(w, 0.0, gain),
    "uniform": lambda w, gain, a, mode: nn.init.uniform_(w, -gain, gain),
    "xavier_uniform": lambda w, gain, a, mode: nn.init.xavier_uniform_(w, gain=gain),
    "xavier_normal": lambda w, gain, a, mode: nn.init.xavier_normal_(w, gain=gain),
    "orthogonal": lambda w, gain, a, mode: nn.init.orthogonal_(w, gain=gain),
}

_NONLINEARITIES = {
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False),
    "relu": lambda: nn.ReLU(inplace=False),
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "silu": lambda: nn.SiLU(inplace=False),
    "swish": lambda: nn.SiLU(inplace=False),
}


def init_weight(
    weight: torch.Tensor,
    w_init_gain: str = "linear",
    init_method: str = "xavier_normal",
    lrelu_nslope: float = 0.01,
    kaiming_fan_mode: str = "fan_in",
):
    """Initialize a linear or convolutional weight tensor in place.

    ``w_init_gain`` names the nonlinearity that follows the layer, and is used to
    look up a gain through ``torch.nn.init.calculate_gain``. SiLU has no gain of
    its own; it borrows ReLU's, which its shape closely follows.

    Supported ``init_method`` values are normal, uniform, xavier_uniform,
    xavier_normal (alias xavier), kaiming_uniform, kaiming_normal (alias
    kaiming), and orthogonal.
    """
    gain_name = _GAIN_ALIASES.get(w_init_gain, w_init_gain)
    method = {"xavier": "xavier_normal", "kaiming": "kaiming_normal"}.get(init_method, init_method)

    if gain_name == "leaky_relu":
        gain = nn.init.calculate_gain(gain_name, lrelu_nslope)
        kaiming_slope = lrelu_nslope
    else:
        gain = nn.init.calculate_gain(gain_name)
        kaiming_slope = 0.0

    if method.startswith("kaiming"):
        if gain_name not in {"relu", "leaky_relu"}:
            warnings.warn(
                f"kaiming initialization is derived for rectifiers; using it with "
                f"'{w_init_gain}' is not recommended."
            )
        with torch.no_grad():
            fn = nn.init.kaiming_uniform_ if method == "kaiming_uniform" else nn.init.kaiming_normal_
            fn(weight, a=kaiming_slope, mode=kaiming_fan_mode, nonlinearity=gain_name)
        return

    try:
        initializer = _INIT_METHODS[method]
    except KeyError:
        raise NotImplementedError(f"initialization method [{init_method}] is not implemented")
    with torch.no_grad():
        initializer(weight, gain, kaiming_slope, kaiming_fan_mode)


class LinearLayer(nn.Linear):
    """``nn.Linear`` with configurable initialization and two optional extras.

    ``lr_multiplier`` rescales weight and bias in the forward pass, so the layer
    effectively learns at that fraction of the optimizer's rate. ``fixed_bias``
    adds a constant to the output that is not a parameter.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_init_gain: str = "linear",
        init_method: str = "xavier_normal",
        lrelu_nslope: float = 0.01,
        kaiming_fan_mode: str = "fan_in",
        bias_init_val: float = 0.0,
        lr_multiplier: float = 1.0,
        fixed_bias: float = None,
    ):
        super().__init__(in_features, out_features, bias)
        init_weight(
            self.weight,
            w_init_gain=w_init_gain,
            init_method=init_method,
            lrelu_nslope=lrelu_nslope,
            kaiming_fan_mode=kaiming_fan_mode,
        )
        if self.bias is not None:
            nn.init.constant_(self.bias, bias_init_val)
        self.lr_multiplier = lr_multiplier
        self.fixed_bias = fixed_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.lr_multiplier
        bias = None if self.bias is None else self.bias * scale
        out = F.linear(input, self.weight * scale, bias)
        if self.fixed_bias is not None:
            out = out + self.fixed_bias
        return out


class StackedLinearLayers(nn.Module):
    """A stack of ``num_layers`` linear layers with nonlinearities between them.

    Each block is ordered Linear -> normalization -> nonlinearity -> dropout. The
    final linear layer is initialized for a linear output and, unless
    ``output_add_nonlinearity`` is set, is followed by nothing.

    ``dim_features`` is either one width shared by every hidden layer, or a
    sequence of ``num_layers - 1`` widths. When a normalization layer is added,
    the preceding linear layer drops its bias, which the normalization supplies.
    """

    def __init__(
        self,
        num_layers: int,
        dim_input: int,
        dim_output: int,
        dim_features: T.Union[T.Sequence[int], int],
        nonlinearity: str = "leaky_relu",
        add_norm_layer: bool = False,
        norm_fun: T.Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
        output_add_nonlinearity: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.add_norm_layer = add_norm_layer
        self.linear_bias = not add_norm_layer
        self.norm_fun = norm_fun
        self.nonlinearity = nonlinearity
        self.output_add_nonlinearity = output_add_nonlinearity

        self.dim_features = self._hidden_widths(dim_features, num_layers)

        self.dropout_prob = dropout_prob
        if dropout_prob > 0 and num_layers == 1:
            self.dropout_prob = 0.0
            warnings.warn(
                "dropout is inserted after every layer but the last, so it has no effect "
                f"with num_layers=1 (got dropout={dropout_prob})"
            )

        if nonlinearity not in _NONLINEARITIES:
            raise ValueError(f"unsupported nonlinearity '{nonlinearity}'")
        make_activation = _NONLINEARITIES[nonlinearity]

        # Channel-first normalizations need (batch, feature, seq) rather than
        # (batch, seq, feature); forward() transposes when one is present.
        self.permute_for_norm = False

        self.main = nn.ModuleList()
        self.addons = nn.ModuleList()

        width = dim_input
        for hidden_width in self.dim_features:
            self.main.append(
                LinearLayer(
                    in_features=width,
                    out_features=hidden_width,
                    bias=self.linear_bias,
                    w_init_gain=nonlinearity,
                )
            )
            width = hidden_width
            self.addons.append(self._make_addon(width, make_activation))

        self.main.append(
            LinearLayer(in_features=width, out_features=dim_output, bias=True, w_init_gain="linear")
        )
        if output_add_nonlinearity:
            self.addons.append(self._make_addon(dim_output, make_activation))

    @staticmethod
    def _hidden_widths(dim_features, num_layers):
        if isinstance(dim_features, int):
            return [dim_features] * (num_layers - 1)
        if isinstance(dim_features, T.Sequence):
            widths = list(dim_features)
            if len(widths) == 1:
                return widths * (num_layers - 1)
            if len(widths) != num_layers - 1 and num_layers != 1:
                raise ValueError(
                    f"dim_features has {len(widths)} entries, expected {num_layers - 1}"
                )
            return widths
        raise ValueError(f"dim_features must be an int or a sequence, got {type(dim_features)}")

    def _make_addon(self, width, make_activation):
        layers = []
        if self.add_norm_layer:
            norm = self.norm_fun(width)
            layers.append(norm)
            if isinstance(norm, (nn.BatchNorm1d, nn.InstanceNorm1d)):
                self.permute_for_norm = True
            elif not isinstance(norm, nn.LayerNorm):
                warnings.warn(
                    f"unrecognized normalization {norm}; check whether its input needs "
                    "the feature dimension moved before the sequence dimension"
                )
        layers.append(make_activation())
        if self.dropout_prob > 0:
            layers.append(nn.Dropout(p=self.dropout_prob, inplace=False))
        return nn.Sequential(*layers)

    def _apply_addon(self, x, addon):
        transpose = self.permute_for_norm and x.dim() == 3
        if transpose:
            x = x.permute(0, 2, 1)
        x = addon(x)
        if transpose:
            x = x.permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, addon in zip(self.main[:-1], self.addons):
            x = self._apply_addon(linear(x), addon)
        x = self.main[-1](x)
        if self.output_add_nonlinearity:
            x = self._apply_addon(x, self.addons[-1])
        return x
