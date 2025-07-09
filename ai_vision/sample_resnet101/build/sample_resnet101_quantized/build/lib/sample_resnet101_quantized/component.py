from rclpy.components import Component
from sample_resnet101_quantized.qrb_ros_resnet101 import ResNet101QuantizedNode

class SampleComponent(Component):
    def __init__(self):
        super().__init__(name='sample_component', node_class='ResNet101QuantizedNode')

