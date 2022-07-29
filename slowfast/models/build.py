#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import functools
import torch
from fvcore.common.registry import Registry
from torch.distributed.algorithms.ddp_comm_hooks import (
    default as comm_hooks_default,
)
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn import auto_wrap, enable_wrap, default_auto_wrap_policy

import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.utils.distributed as du

logger = logging.get_logger(__name__)

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def get_fsdp_params(cfg) -> dict:
    """
    Return FSDP params dict (Assumes usage of fairscale FSDP)

    Args:
        cfg: config containing FSDP wrapping params

    Return:
        fsdp_params: FSDP params dict
    """
    fsdp_params = {}
    try:
        fsdp_params['reshard_after_forward'] = cfg.FSDP.RESHARD_AFTER_FW
        fsdp_params["mixed_precision"] = cfg.TRAIN.MIXED_PRECISION
    except AttributeError as e:
        logger.exception(f"Configuration error: {e}")
        raise e
    if cfg.FSDP.AUTO_WRAP:
        fsdp_params['auto_wrap_policy'] = functools.partial(
            default_auto_wrap_policy,
            min_num_params=int(cfg.FSDP.MIN_PARAMS_TO_WRAP)
        )

    return fsdp_params


def fsdp_model(model: torch.nn.Module, cfg: dict, cur_device: torch.device):
    """
    Wraps a model with FSDP.

    Args:
        model: unwrapped torch model
        cfg: config containing FSDP config params
        cur_device: The device to which the model should be transfered

    Return:
        model: FSDP wrapped model
    """
    assert not (cfg.FSDP.AUTO_WRAP and cfg.FSDP.NESTED_WRAP), "FSDP mode: select either AUTO_WRAP or NESTED_WRAP"
    fsdp_params = get_fsdp_params(cfg)

    if cfg.FSDP.AUTO_WRAP:
        with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
            model = auto_wrap(model)
    elif(cfg.FSDP.NESTED_WRAP):
        model_name = cfg.MODEL.MODEL_NAME
        with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
            model = MODEL_REGISTRY.get(model_name)(cfg)

    model = FSDP(
            model,
            reshard_after_forward=cfg.FSDP.RESHARD_AFTER_FW,
            mixed_precision=cfg.TRAIN.MIXED_PRECISION,
        )

    model = model.cuda(cur_device)
    return model


def ddp_model(model, cfg, cur_device):
    """
    Wraps a model with DDP.

    Args:
        model: unwrapped torch model
        cfg: global config
        cur_device: The device to which the model should be transfered

    Return:
        model: DDP wrapped model
    """
    # Make model replica operate on the current device
    model = model.cuda(cur_device)
    model = torch.nn.parallel.DistributedDataParallel(
        module=model,
        device_ids=[cur_device],
        output_device=cur_device,
        find_unused_parameters=True
        if cfg.MODEL.DETACH_FINAL_FC
        or cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        else False,
    )
    if cfg.MODEL.FP16_ALLREDUCE:
        model.register_comm_hook(
            state=None, hook=comm_hooks_default.fp16_compress_hook
        )
    return model


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    cur_device = 'cpu'
    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id

    model_name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(model_name)(cfg)
    if cfg.BN.NORM_TYPE == "sync_batchnorm_apex":
        try:
            import apex
        except ImportError:
            raise ImportError("APEX is required for this model, pelase install")

        logger.info("Converting BN layers to Apex SyncBN")
        process_group = apex.parallel.create_syncbn_process_group(
            group_size=cfg.BN.NUM_SYNC_DEVICES
        )
        model = apex.parallel.convert_syncbn_model(
            model, process_group=process_group
        )

    # NOTE: jit analysis will report incorrect FLOPS if activation ckpt is enabled
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True, device='cpu')

    if cfg.NUM_GPUS > 1:
        # Use multi-process data parallel model in the multi-gpu setting
        if cfg.FSDP.ENABLED:
            model = fsdp_model(model, cfg, cur_device)
        else:
            model = ddp_model(model, cfg, cur_device)

    return model
