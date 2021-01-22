r""" Various functions, used for manual testing
"""
import torch
import torch.nn.functional as F
from model import (Backbone,
                   TaskClassifier, MultiHeadClassifier, ChannelGatedCL)


# TODO: use some proper unittest package, not just plain functions
def test_ChannelGatedCL_output_shape():
    net = ChannelGatedCL(N_tasks=10, in_ch=1, out_dim=2)
    input = torch.rand(10, 1, 16, 16)
    data = {'data': input,
            'head_idx': 1}
    net.eval()
    out, task_idx = net(data)
    assert F.softmax(out, dim=-1).shape == (10, 2)


def test_Backbone_output_shape():
    bb = Backbone(2, 3, 100)
    input = torch.rand(1, 3, 16, 16)
    assert bb(input).shape == (1, 2, 100, 2, 2)


def test_TaskClassifier_output_shape():
    tc = TaskClassifier(3, 100)
    input = torch.rand(4, 3, 100, 16, 16)
    assert F.softmax(tc(input), dim=-1).shape == (4, 3)


def test_MultiHeadClassifier_output_shape():
    mhc = MultiHeadClassifier(2, 100, 2)
    input = torch.rand(4, 1, 100, 2, 2)
    assert F.softmax(mhc(input, torch.LongTensor([0])), dim=-1).shape == (1, 2)


def test_FreezableConv2d_freezing():
    from src.freezable_layers import FreezableConv2d
    fc = FreezableConv2d(3, 5, 3)
    input = torch.ones((32, 3, 12, 12))
    old_w, old_b = fc.frozen_weight, fc.frozen_bias
    fc.freeze(torch.LongTensor([1, 1, 0, 0, 1]))
    loss = fc(input).sum()
    loss.backward()
    fc.weight.data -= 1e-2 * fc.weight.grad
    fc.bias.data -= 1e-2 * fc.bias.grad

    assert torch.allclose(old_w[[0, 1, -1]].data, fc.frozen_weight[[0, 1, -1]].data)
    assert torch.allclose(old_b[[0, 1, -1]].data, fc.frozen_bias[[0, 1, -1]].data)


def test_FreezableConv2d_reinit():
    from src.freezable_layers import FreezableConv2d
    fc = FreezableConv2d(3, 5, 3)
    input = torch.ones((32, 3, 12, 12))
    old_w, old_b = fc.frozen_weight, fc.frozen_bias
    fc.freeze(torch.LongTensor([1, 1, 0, 0, 1]))
    fc.reinit_unfrozen()
    assert not torch.allclose(old_w[[2, 3]].data, fc.frozen_weight[[2, 3]].data)
    assert not torch.allclose(old_b[[2, 3]].data, fc.frozen_bias[[2, 3]].data)


def bn_forgetting_debug():
    import torch

    from src.data import prepare_dataset, get_loaders, set_task
    from config import cfg
    from src.model import ChannelGatedCL
    from src.visualization import extract_taskwise_frozen_masks
    from src.freezable_layers import FreezableBatchNorm2d
    # Get data
    datasets = prepare_dataset(dataset_name=cfg.DATASET_NAME,
                               task_pairs=cfg.TASK_PAIRS,
                               truncate_size=cfg.TRUNCATE_SIZE)
    dataloaders = get_loaders(*datasets, batch_size=cfg.BATCH_SIZE)
    train_data, val_data, test_data = datasets
    train_loader, val_loader, test_loader = dataloaders
    set_task(0, train_data, val_data, test_data)

    batch = next(iter(test_loader))
    img, label, head_idx = (batch[0][:], batch[1][:], batch[2][:])

    # Initiate the model
    CGCL = ChannelGatedCL(in_ch=cfg.IN_CH, out_dim=cfg.OUT_DIM,
                          conv_ch=cfg.CONV_CH,
                          sparsity_patience_epochs=cfg.SPARSITY_PATIENCE_EPOCHS,
                          lambda_sparse=cfg.LAMBDA_SPARSE,
                          freeze_fixed_proc=cfg.FREEZE_FIXED_PROC,
                          freeze_prob_thr=cfg.FREEZE_PROB_THR)
    CGCL.eval()

    task_of_interest = 1
    layer_of_interest = 3
    channel_of_interest = 1 #112
    frozen_masks = extract_taskwise_frozen_masks(cfg.RESULTS_ROOT)
    fr_idx = frozen_masks[task_of_interest][layer_of_interest].nonzero(as_tuple=False).squeeze()
    print('channel_of_interest is frozen for tasks > task_of_interest:', channel_of_interest in fr_idx)
    for i in range(0, 5):
        #     ones = torch.ones_like(img)
        expanded_data = img[:, None, :, :, :].expand(-1, i + 1, -1, -1, -1).clone().cuda()
        CGCL.load_model_state_dict(f'after_task_{i}.ckpt')
        # by default, models are loaded in training_mode; if required - switch to eval mode manually
        CGCL.eval()
        CGCL.cuda()
        #     print([f'{CGCL.backbone.layers(expanded_data)[i, 0].mean().item():.4f}' for i in [6]])
        debug_preproc = True
        repr_data = CGCL.backbone.layers[:layer_of_interest](expanded_data)
        debug_preproc = False
        repeated_data = torch.stack([repr_data[0], repr_data[0]], dim=0)
        #     print(CGCL.backbone.layers[0][0].fbn.shadow_weight)
        #     print(repr_data[0].nonzero(as_tuple=True)[0].unique())
        #     sys.stdout.flush()

        if i < task_of_interest:
            CGCL.add_task()
            continue
        if i == task_of_interest:
            # CGCL.backbone.layers[layer_of_interest][0].fbns_main2[1] = torch.nn.Identity()
            old_out = CGCL.backbone.layers[layer_of_interest](repr_data)[:, 1].clone()
            old_gate_logits = CGCL.backbone.layers[layer_of_interest][0].gates[1](repr_data[:, 1])[0].clone()
            old_mask = CGCL.backbone.layers[layer_of_interest][0].sample_channels_mask(old_gate_logits)
            # old = CGCL.backbone.layers[layer_of_interest][0].fbns_main1[1].shadow_running_var[20]
            #         print(old_out[20])
            # CGCL.backbone.layers[layer_of_interest][0].fbns_main2[1] = FreezableBatchNorm2d(128)
            CGCL.add_task()
            continue
        else:
            # CGCL.backbone.layers[layer_of_interest][0].fbns_main2[1] = torch.nn.Identity()
            new_out = CGCL.backbone.layers[layer_of_interest](repr_data)[:, 1].clone()
            new_gate_logits = CGCL.backbone.layers[layer_of_interest][0].gates[1](repr_data[:, 1])[0].clone()
            new_mask = CGCL.backbone.layers[layer_of_interest][0].sample_channels_mask(new_gate_logits)

            # new = CGCL.backbone.layers[layer_of_interest][0].fbns_main1[1].shadow_running_var[20]
            #         print(new)
            # print(torch.allclose(old_out[channel_of_interest], new_out[channel_of_interest], atol=1e-7, rtol=0))
            """Check changed channels"""
            print((1 - torch.isclose(old_out[29], new_out[29], atol=1e-7, rtol=0).long()).nonzero(as_tuple=True)[0].unique())
            # print(new_out.nonzero(as_tuple=True)[0].unique())
            old_out = new_out.clone()
            # CGCL.backbone.layers[layer_of_interest][0].fbns_main2[1] = FreezableBatchNorm2d(128)
        #     print(torch.conv2d(expanded_data[6, 0].unsqueeze(0), CGCL.backbone.layers[0][0].conv2d.weight[0].unsqueeze(0).data))
        #     logits, _ = CGCL(img, head_idx)
        #     print(logits[6])
        CGCL.add_task()
    #     CGCL.model.multihead_clf
if __name__ == '__main__':
    bn_forgetting_debug()
