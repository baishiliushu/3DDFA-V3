import argparse

import torch
import sys

from model.recon import face_model

import onnxruntime


#python demo.py --inputpath /work/project/Gerry/data/face_data/face/box_0604/sum_data/ 
#--savepath /work/project/Gerry/data/face_data/face/box_0604/sum_data_result --device cuda 
#--iscrop 1 --detector retinaface --ldm68 1 --useTex 1 --extractTex 1 --backbone mbnetv3

ARGS_SET = {"device":"cuda",  "ldm68": True,  "useTex": True, "extractTex": True, "backbone": "mbnetv3"}
ONNX_SAVE_PREFIX = "./3ddfa_v3"
OPSET_VERSION_SET = 11

def parser_set(fixed_args=ARGS_SET):
    parser = argparse.ArgumentParser(description='3DDFA-V3')

    parser.add_argument('-i', '--inputpath', default='examples/', type=str,
                        help='path to the test data, should be a image folder')
    parser.add_argument('-s', '--savepath', default='examples/results', type=str,
                        help='path to the output directory, where results (obj, png files) will be stored.')
    parser.add_argument('--device', default=fixed_args["device"], type=str,
                        help='set device, cuda or cpu' )

    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped and resized into (224,224,3).' )
    parser.add_argument('--detector', default='retinaface', type=str,
                        help='face detector for cropping image, support for mtcnn and retinaface')

    # save
    parser.add_argument('--ldm68', default=fixed_args["ldm68"], type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 68 landmarks')
    parser.add_argument('--ldm106', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 106 landmarks')
    parser.add_argument('--ldm106_2d', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 106 landmarks, face profile is in 2d form')
    parser.add_argument('--ldm134', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 134 landmarks' )
    parser.add_argument('--seg', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show segmentation in 2d without visible mask' )
    parser.add_argument('--seg_visible', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show segmentation in 2d with visible mask' )
    parser.add_argument('--useTex', default=fixed_args["useTex"], type=lambda x: x.lower() in ['true', '1'],
                        help='save obj use texture from BFM model')
    parser.add_argument('--extractTex', default=fixed_args["extractTex"], type=lambda x: x.lower() in ['true', '1'],
                        help='save obj use texture extracted from input image')

    # backbone
    parser.add_argument('--backbone', default=fixed_args["backbone"], type=str,
                        help='backbone for reconstruction, support for resnet50 and mbnetv3')
    return parser.parse_args()

# device_type = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torchsummary.summary(model, (3,224,224))

# 加载PyTorch模型
print("{}\n{}".format(type(parser_set()), parser_set()))

net = face_model(parser_set())  
model = net.net_recon
print("********************if nn_model :{}**********************".format(isinstance(model, torch.nn.Module)))
print(model)
print("****************************************************")
# 获取模型输入
"""
size_input_src = 640
dummy_input_src = np.zeros((size_input_src, size_input_src, 3), dtype=np.uint8)
alpha = model.net_recon(self.input_img)
alpha_dict = model.split_alpha(alpha)
face_shape = model.compute_shape(alpha_dict['id'], alpha_dict['exp'])
"""

# 生成随机输入（根据模型输入尺寸调整）
batch_size = 1
size_input = 224 # https://github.com/wang-zidu/3DDFA-V3/blob/main/util/preprocess.py -> align_img()
dummy_input = torch.randn(batch_size, 3, size_input, size_input).cuda() 

onnx_full_name = "{}-{}_opset{}.onnx".format(ONNX_SAVE_PREFIX, ARGS_SET["backbone"], OPSET_VERSION_SET)
# 导出ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_full_name,
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=True,
    opset_version=OPSET_VERSION_SET
    )
print("Convert to {} done.".format(onnx_full_name))


# 3. 推理前转换：CUDA Tensor -> CPU NumPy
onnx_inputs = {"input": dummy_input.cpu().numpy()} 
session = onnxruntime.InferenceSession(onnx_full_name, None)
outputs = session.run(None, onnx_inputs)

print("Run {} output shape : {} v.s. (1, 257)".format(onnx_full_name, outputs[0].shape))

#providers=['CPUExecutionProvider']/providers=['CUDAExecutionProvider'] ; onnxruntime-gpu/onnxruntime
"""
/home/ljx/.virtualenvs/pytorch18/lib/python3.7/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:353: UserWarning: Deprecation warning. This ORT build has ['CUDAExecutionProvider', 'CPUExecutionProvider'] enabled. The next release (ORT 1.10) will require explicitly setting the providers parameter (as opposed to the current behavior of providers getting set/registered by default based on the build flags) when instantiating InferenceSession.For example, onnxruntime.InferenceSession(..., providers=["CUDAExecutionProvider"], ...)
  "based on the build flags) when instantiating InferenceSession."
"""
