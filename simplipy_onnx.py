# -*- coding: utf-8 -*-

import onnx
from onnxsim import simplify

# python3 -m onnxsim ./3ddfa_v3-mbnetv3_opset11.onnx ./3ddfa_v3-mbnetv3_opset11-simplify.onnx

ONNX_FILE_NAME = "./3ddfa_v3-mbnetv3_opset11"

sim_full_name = "{}-simplify.onnx".format(ONNX_FILE_NAME)
simplified_model, check = simplify(onnx_full_name, skip_fuse_bn=True)
onnx.save_model(simplified_model, sim_full_name)
onnx.checker.check_model(simplified_model)
if check is True:
    print("模型简化成功")
else:
    print("简化模型失败")
