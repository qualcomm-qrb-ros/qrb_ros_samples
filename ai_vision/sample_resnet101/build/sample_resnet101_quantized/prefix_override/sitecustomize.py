import sys
if sys.prefix == '/local/mnt/workspace/fulan/test/qirp-sdk/toolchain/install_dir/sysroots/x86_64-qcomsdk-linux/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/ai_vision/sample_resnet101_quantized/install/sample_resnet101_quantized'
