from unittest import TestCase
from spair.modules import Backbone

class TestBackbone(TestCase):

    def test_compute_output_shape(self):
        self.backbone = Backbone(input_shape=(32, 3, 128, 128), n_out_channels=100)
        self.backbone.compute_output_shape()
        self.fail()

    def test__build_backbone(self):
        self.fail()

    def test__build_receptive_field_padding(self):
        self.fail()

    def test_forward(self):
        self.fail()
