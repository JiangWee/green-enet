# ...existing code...
import torch
import torch.nn as nn
from models.enet import ENet
from collections import OrderedDict

def convert_to_onnx():
    # 加载训练好的模型或 state_dict
    ckpt_path = r'D:\code\PyTorch-ENet-master\PyTorch-ENet-master\save\ENet_CamVid\ENet'
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        # checkpoint 可能是 {'state_dict': ...} 或 直接是 state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 根据你的 ENet 构造函数调整参数（例如 num_classes）
        try:
            model = ENet()
        except TypeError:
            model = ENet(num_classes=12)  # 若 ENet 需要 num_classes，请改为正确值

        # 处理可能的 'module.' 前缀（DataParallel 保存时会有）
        new_state = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state[name] = v
        model.load_state_dict(new_state, strict=False)
    else:
        # checkpoint 已经是完整的 model 对象
        model = checkpoint

    model.eval()
    model.to('cpu')


    dummy_input = torch.randn(1, 3, 360, 480)

    # 强制使用纯 ONNX 导出（不使用 ATEN fallback）
    tried = False
    OpTypes = getattr(torch.onnx, 'OperatorExportTypes', None)
    if OpTypes is None:
        print("无法找到 torch.onnx.OperatorExportTypes，导出将失败。")
    else:
        for opset in (11, 12, 13, 14):
            try:
                export_kwargs = dict(
                    model=model,
                    args=dummy_input,
                    f=f"enet_model_opset{opset}.onnx",
                    input_names=['input'],
                    output_names=['output'],
                    opset_version=opset,
                    operator_export_type=OpTypes.ONNX,
                    do_constant_folding=True
                )
                # older torch versions may not accept some kwargs (e.g. add_node_names)
                torch.onnx.export(**export_kwargs)
                print(f"ENet 模型纯 ONNX 导出成功（opset={opset}） -> enet_model_opset{opset}.onnx")
                tried = True
                break
            except RuntimeError as e:
                print(f"纯 ONNX 导出失败（opset={opset}）：{e}")
            except TypeError as e:
                # 捕获不支持的参数等类型错误并报告
                print(f"导出时发生 TypeError（opset={opset}）：{e}")
                break

    if not tried:
        print("所有纯 ONNX 导出尝试失败。若需要，改回使用 ATEN fallback 或排查具体不支持的算子。")

if __name__ == "__main__":
    convert_to_onnx()
# ...existing code...