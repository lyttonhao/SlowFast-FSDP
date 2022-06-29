#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import av
import decord as de
de.bridge.set_bridge('torch')

import torch

def get_video_container(path_to_vid, multi_thread_decode=False, backend="pyav", device = 'cpu'):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """
    if backend == "torchvision":
        with open(path_to_vid, "rb") as fp:
            container = fp.read()
        return container
    elif backend == "pyav":
        container = av.open(path_to_vid)
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        return container
    elif backend == "decord":
        worker_id = torch.utils.data.get_worker_info().id
        if device == 'cpu':
            NUM_CPUS = os.cpu_count()
            ctx = de.cpu(worker_id % NUM_CPUS)
        else:
            #NOTE: Install decord from source to support GPU acceleration. Requires some apt pkgs !
            MAX_GPUS_PER_NODE = torch.cuda.device_count()
            ctx = de.gpu(worker_id % MAX_GPUS_PER_NODE)
        container = de.VideoReader(path_to_vid, ctx = ctx)
        return container
    else:
        raise NotImplementedError("Unknown backend {}".format(backend))