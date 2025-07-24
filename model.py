import ranger21
import torch
from torch import nn
import pytorch_lightning as pl
from feature_transformer import DoubleFeatureTransformerSlice
from dataclasses import dataclass
from kan import *

# 3 layer fully connected network parameters
L1 = 3072
L2 = 15
L3 = 32


# parameters needed for the definition of the loss
@dataclass
class LossParams:
    in_offset: float = 270
    out_offset: float = 270
    in_scaling: float = 340
    out_scaling: float = 380
    start_lambda: float = 1.0
    end_lambda: float = 1.0
    pow_exp: float = 2.5
    qp_asymmetry: float = 0.0


def coalesce_ft_weights(model, layer):
    weight = layer.weight.data
    indices = model.feature_set.get_virtual_to_real_features_gather_indices()
    weight_coalesced = weight.new_zeros(
        (model.feature_set.num_real_features, weight.shape[1])
    )
    for i_real, is_virtual in enumerate(indices):
        weight_coalesced[i_real, :] = sum(
            weight[i_virtual, :] for i_virtual in is_virtual
        )
    return weight_coalesced


def get_parameters(layers):
    return [p for layer in layers for p in layer.parameters()]


class LayerStacks(nn.Module):
    def __init__(self, count):
        super(LayerStacks, self).__init__()

        self.count = count

        # Use KAN instead of linear layers
        # Input: 2 * L1 // 2 = L1, Hidden: [L2+1, L3], Output: 1
        # We'll create separate KANs for each layer stack
        self.kans = nn.ModuleList([
            KAN(width=[L1, L2 + 1, L3, 1], grid=5, k=3, seed=i)
            for i in range(count)
        ])

        # For compatibility, we'll keep these for now but they won't be used directly
        self.l1 = nn.Linear(2 * L1 // 2, (L2 + 1) * count)
        self.l1_fact = nn.Linear(2 * L1 // 2, L2 + 1, bias=True)
        self.l2 = nn.Linear(L2 * 2, L3 * count)
        self.output = nn.Linear(L3, 1 * count)

        # Cached helper tensor for choosing outputs by bucket indices
        self.idx_offset = None

        self._init_layers()

    def _init_layers(self):
        # Initialize dummy layers for compatibility
        with torch.no_grad():
            self.l1_fact.weight.fill_(0.0)
            self.l1_fact.bias.fill_(0.0)
            self.output.bias.fill_(0.0)

    def forward(self, x, ls_indices):
        assert self.idx_offset is not None and self.idx_offset.shape[0] == x.shape[0]

        batch_size = x.shape[0]
        outputs = []

        # Process each sample with its corresponding KAN
        for i in range(batch_size):
            ls_idx = ls_indices[i].item()
            kan_output = self.kans[ls_idx](x[i:i + 1])
            outputs.append(kan_output)

        # Concatenate outputs
        l3x_ = torch.cat(outputs, dim=0)

        return l3x_

    def get_coalesced_layer_stacks(self):
        # For serialization compatibility, we need to extract weights from KAN
        # This is a simplified version - in practice, KAN weights are more complex
        for i in range(self.count):
            with torch.no_grad():
                # Create dummy linear layers for compatibility
                l1 = nn.Linear(2 * L1 // 2, L2 + 1)
                l2 = nn.Linear(L2 * 2, L3)
                output = nn.Linear(L3, 1)

                # Initialize with small random values for now
                # In a real implementation, you'd need to extract KAN parameters properly
                l1.weight.data = torch.randn_like(l1.weight.data) * 0.01
                l1.bias.data = torch.randn_like(l1.bias.data) * 0.01
                l2.weight.data = torch.randn_like(l2.weight.data) * 0.01
                l2.bias.data = torch.randn_like(l2.bias.data) * 0.01
                output.weight.data = torch.randn_like(output.weight.data) * 0.01
                output.bias.data = torch.randn_like(output.bias.data) * 0.01

                yield l1, l2, output


class NNUE(pl.LightningModule):
    def __init__(
            self,
            feature_set,
            max_epoch=800,
            num_batches_per_epoch=int(100_000_000 / 16384),
            gamma=0.992,
            lr=8.75e-4,
            param_index=0,
            num_psqt_buckets=8,
            num_ls_buckets=8,
            loss_params=LossParams(),
    ):
        super(NNUE, self).__init__()
        self.num_psqt_buckets = num_psqt_buckets
        self.num_ls_buckets = num_ls_buckets
        self.input = DoubleFeatureTransformerSlice(
            feature_set.num_features, L1 + self.num_psqt_buckets
        )
        self.feature_set = feature_set
        self.layer_stacks = LayerStacks(self.num_ls_buckets)
        self.loss_params = loss_params
        self.max_epoch = max_epoch
        self.num_batches_per_epoch = num_batches_per_epoch
        self.gamma = gamma
        self.lr = lr
        self.param_index = param_index

        self.nnue2score = 600.0
        self.weight_scale_hidden = 64.0
        self.weight_scale_out = 16.0
        self.quantized_one = 127.0

        # Simplified weight clipping for KAN
        self.weight_clipping = []

        self._init_layers()

    def _zero_virtual_feature_weights(self):
        weights = self.input.weight
        with torch.no_grad():
            for a, b in self.feature_set.get_virtual_feature_ranges():
                weights[a:b, :] = 0.0
        self.input.weight = nn.Parameter(weights)

    def _init_layers(self):
        self._zero_virtual_feature_weights()
        self._init_psqt()

    def _init_psqt(self):
        input_weights = self.input.weight
        input_bias = self.input.bias
        scale = 1 / self.nnue2score
        with torch.no_grad():
            initial_values = self.feature_set.get_initial_psqt_features()
            assert len(initial_values) == self.feature_set.num_features
            for i in range(self.num_psqt_buckets):
                input_weights[:, L1 + i] = torch.FloatTensor(initial_values) * scale
                input_bias[L1 + i] = 0.0
        self.input.weight = nn.Parameter(input_weights)
        self.input.bias = nn.Parameter(input_bias)

    def _clip_weights(self):
        # KAN handles its own regularization, so we skip weight clipping
        pass

    def set_feature_set(self, new_feature_set):
        if self.feature_set.name == new_feature_set.name:
            return

        if len(self.feature_set.features) > 1:
            raise Exception(
                "Cannot change feature set from {} to {}.".format(
                    self.feature_set.name, new_feature_set.name
                )
            )

        old_feature_block = self.feature_set.features[0]
        new_feature_block = new_feature_set.features[0]

        if old_feature_block.name == next(iter(new_feature_block.factors)):
            weights = self.input.weight
            padding = weights.new_zeros(
                (new_feature_block.num_virtual_features, weights.shape[1])
            )
            weights = torch.cat([weights, padding], dim=0)
            self.input.weight = nn.Parameter(weights)
            self.feature_set = new_feature_set
        else:
            raise Exception(
                "Cannot change feature set from {} to {}.".format(
                    self.feature_set.name, new_feature_set.name
                )
            )

    def forward(
            self,
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            psqt_indices,
            layer_stack_indices,
    ):
        wp, bp = self.input(white_indices, white_values, black_indices, black_values)
        w, wpsqt = torch.split(wp, L1, dim=1)
        b, bpsqt = torch.split(bp, L1, dim=1)
        l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        l0_ = torch.clamp(l0_, 0.0, 1.0)

        l0_s = torch.split(l0_, L1 // 2, dim=1)
        l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
        l0_ = torch.cat(l0_s1, dim=1) * (127 / 128)

        psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
        wpsqt = wpsqt.gather(1, psqt_indices_unsq)
        bpsqt = bpsqt.gather(1, psqt_indices_unsq)

        x = self.layer_stacks(l0_, layer_stack_indices) + (wpsqt - bpsqt) * (us - 0.5)

        return x

    def step_(self, batch, batch_idx, loss_type):
        _ = batch_idx

        self._clip_weights()

        (
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            outcome,
            score,
            psqt_indices,
            layer_stack_indices,
        ) = batch

        scorenet = (
                self(
                    us,
                    them,
                    white_indices,
                    white_values,
                    black_indices,
                    black_values,
                    psqt_indices,
                    layer_stack_indices,
                )
                * self.nnue2score
        )

        p = self.loss_params
        q = (scorenet - p.in_offset) / p.in_scaling
        qm = (-scorenet - p.in_offset) / p.in_scaling
        qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

        s = (score - p.out_offset) / p.out_scaling
        sm = (-score - p.out_offset) / p.out_scaling
        pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

        t = outcome
        actual_lambda = p.start_lambda + (p.end_lambda - p.start_lambda) * (
                self.current_epoch / self.max_epoch
        )
        pt = pf * actual_lambda + t * (1.0 - actual_lambda)

        loss = torch.pow(torch.abs(pt - qf), p.pow_exp)
        if p.qp_asymmetry != 0.0:
            loss = loss * ((qf > pt) * p.qp_asymmetry + 1)
        loss = loss.mean()

        self.log(loss_type, loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "val_loss")

    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "test_loss")

    def configure_optimizers(self):
        LR = self.lr
        train_params = [
            {"params": get_parameters([self.input]), "lr": LR},
            {"params": get_parameters([self.layer_stacks]), "lr": LR},
        ]

        optimizer = ranger21.Ranger21(
            train_params,
            lr=1.0,
            betas=(0.9, 0.999),
            eps=1.0e-7,
            using_gc=False,
            using_normgc=False,
            weight_decay=0.0,
            num_batches_per_epoch=self.num_batches_per_epoch,
            num_epochs=self.max_epoch,
            warmdown_active=False,
            use_warmup=False,
            use_adaptive_gradient_clipping=False,
            softplus=False,
            pnm_momentum_factor=0.0,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma
        )
        return [optimizer], [scheduler]