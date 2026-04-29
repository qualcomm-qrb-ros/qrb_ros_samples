#!/usr/bin/env python3
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from lib.network import PoseNet, PoseRefineNet


LOGGER = logging.getLogger("densefusion_export_onnx")


def unwrap_dataparallel(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.DataParallel):
        return module.module
    for name, child in module.named_children():
        setattr(module, name, unwrap_dataparallel(child))
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DenseFusion models to ONNX")
    parser.add_argument("--model", type=str, required=True, help="PoseNet checkpoint path")
    parser.add_argument("--refine_model", type=str, default="", help="PoseRefineNet checkpoint path")
    parser.add_argument("--output_pose_onnx", type=str, default="densefusion_posenet.onnx")
    parser.add_argument("--output_refine_onnx", type=str, default="densefusion_refiner.onnx")
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--num_obj", type=int, default=None, help="Object class count; auto-infer if omitted")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--input_h", type=int, default=80, help="export-time fixed input image height")
    parser.add_argument("--input_w", type=int, default=80, help="export-time fixed input image width")
    return parser.parse_args()


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _strip_module_prefix(state_dict: dict) -> dict:
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_state_dict(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state_dict = checkpoint["model_state_dict"]
        elif all(isinstance(k, str) for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unsupported checkpoint dict format for: {path}")
    else:
        raise ValueError(f"Unsupported checkpoint type for: {path}")
    return _strip_module_prefix(state_dict)


def infer_num_obj_from_posenet_state_dict(state_dict: dict) -> int:
    key = "conv4_c.bias"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    key = "conv4_c.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    raise KeyError("Unable to infer num_obj from PoseNet checkpoint.")


def infer_num_obj_from_refiner_state_dict(state_dict: dict) -> int:
    key = "conv3_t.bias"
    if key in state_dict:
        return int(state_dict[key].shape[0] // 3)
    key = "conv3_t.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0] // 3)
    raise KeyError("Unable to infer num_obj from PoseRefineNet checkpoint.")


def resolve_output_path(path_arg: str, default_filename: str) -> str:
    path = Path(path_arg)
    if path.exists() and path.is_dir():
        return str(path / default_filename)
    return str(path)


def export_posenet(args: argparse.Namespace) -> None:
    pose_state_dict = load_checkpoint_state_dict(args.model)
    inferred_num_obj = infer_num_obj_from_posenet_state_dict(pose_state_dict)
    if args.num_obj is None:
        args.num_obj = inferred_num_obj
        LOGGER.info("Auto-inferred num_obj=%d from PoseNet checkpoint.", args.num_obj)
    elif args.num_obj != inferred_num_obj:
        LOGGER.warning(
            "Provided num_obj=%d but checkpoint implies num_obj=%d. Using provided value.",
            args.num_obj,
            inferred_num_obj,
        )

    output_pose_onnx = resolve_output_path(args.output_pose_onnx, "densefusion_posenet.onnx")

    model = PoseNet(num_points=args.num_points, num_obj=args.num_obj)
    model = model.to("cpu")
    model.load_state_dict(pose_state_dict)
    model = unwrap_dataparallel(model)
    model = model.to("cpu")
    model.eval()

    dummy_img = torch.randn(1, 3, args.input_h, args.input_w, dtype=torch.float32)
    dummy_points = torch.randn(1, args.num_points, 3, dtype=torch.float32)
    dummy_choose = torch.randint(0, args.input_h * args.input_w, (1, 1, args.num_points), dtype=torch.long)
    dummy_obj = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_img, dummy_points, dummy_choose, dummy_obj),
        output_pose_onnx,
        dynamo=True,
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["img", "points", "choose", "obj"],
        output_names=["pred_r", "pred_t", "pred_c", "emb"],
        external_data=False
    )
    LOGGER.info("Exported PoseNet ONNX: %s", output_pose_onnx)


def export_refiner(args: argparse.Namespace) -> None:
    if not args.refine_model:
        LOGGER.info("Skip PoseRefineNet export, no --refine_model provided.")
        return
    refiner_state_dict = load_checkpoint_state_dict(args.refine_model)
    inferred_num_obj = infer_num_obj_from_refiner_state_dict(refiner_state_dict)
    if args.num_obj is None:
        args.num_obj = inferred_num_obj
        LOGGER.info("Auto-inferred num_obj=%d from PoseRefineNet checkpoint.", args.num_obj)
    elif args.num_obj != inferred_num_obj:
        LOGGER.warning(
            "Provided num_obj=%d but refiner checkpoint implies num_obj=%d. Using provided value.",
            args.num_obj,
            inferred_num_obj,
        )

    output_refine_onnx = resolve_output_path(args.output_refine_onnx, "densefusion_refiner.onnx")

    model = PoseRefineNet(num_points=args.num_points, num_obj=args.num_obj)
    model = model.to("cpu")
    model.load_state_dict(refiner_state_dict)
    model = unwrap_dataparallel(model)
    model = model.to("cpu")
    model.eval()

    dummy_points = torch.randn(1, args.num_points, 3, dtype=torch.float32)
    dummy_emb = torch.randn(1, 32, args.num_points, dtype=torch.float32)
    dummy_obj = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_points, dummy_emb, dummy_obj),
        output_refine_onnx,
        dynamo=True,
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["points", "emb", "obj"],
        output_names=["pred_r", "pred_t"],
        external_data=False
    )
    LOGGER.info("Exported PoseRefineNet ONNX: %s", output_refine_onnx)


def main() -> None:
    """
    /home/data/miniconda3/envs/onepose/bin/python export_onnx.py \
    --model ./trained_checkpoints/linemod/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth \
    --refine_model ./trained_checkpoints/linemod/trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth \
    --num_points 1000 \
    --output_pose_onnx ./ycb-data-onnx-model \
    --output_refine_onnx ./ycb-data-onnx-model
    """
    setup_logger()
    args = parse_args()
    export_posenet(args)
    export_refiner(args)


if __name__ == "__main__":
    main()
