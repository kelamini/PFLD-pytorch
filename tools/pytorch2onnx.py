# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse
from models.pfld import PFLDInference
from torch.autograd import Variable
import torch
import onnxsim

parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--torch_model',
                    default="./checkpoint/snapshot/checkpoint_epoch_191.pth.tar")
parser.add_argument('--onnx_model', default="./output/pfld_191.onnx")
parser.add_argument('--onnx_model_sim',
                    help='Output ONNX model',
                    default="./output/pfld-sim_191.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.torch_model, map_location=torch.device('cpu'))
pfld_backbone = PFLDInference()
pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
print("PFLD bachbone:", pfld_backbone)

print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 3, 112, 112))
input_names = ["input_1"]
output_names = ["output_1"]
torch.onnx.export(pfld_backbone,
                  dummy_input,
                  args.onnx_model,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)

print("====> check onnx model...")
import onnx
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)

print("====> Simplifying...")
model_opt, check = onnxsim.simplify(args.onnx_model)
# print("model_opt", model_opt)
onnx.save(model_opt, args.onnx_model_sim)
print("onnx model simplify Ok!")
