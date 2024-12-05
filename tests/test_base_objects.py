import unittest
import torch

from torch_concepts.base import AnnotatedTensor

class TestAnnotatedTensor(unittest.TestCase):
    def setUp(self):
        self.data = torch.randn(5, 4)
        self.annotations = ["annotation_a", "annotation_b", "annotation_c", "annotation_d"]

    def test_standarize_arguments(self):
        annotations = AnnotatedTensor._standarize_arguments(tensor=self.data, annotations=self.annotations, annotated_axis=1)
        self.assertEqual(annotations, [[], self.annotations])

        annotations = AnnotatedTensor._standarize_arguments(tensor=self.data, annotations=self.annotations, annotated_axis=0)
        self.assertEqual(annotations, [self.annotations, []])

        first_dim_annotations = ["annotation_0", "annotation_1", "annotation_2", "annotation_3", "annotation_4"]
        annotations = AnnotatedTensor._standarize_arguments(tensor=self.data, annotations=[first_dim_annotations, self.annotations], annotated_axis=[0, 1])
        self.assertEqual(annotations, [first_dim_annotations, self.annotations])

        annotations = AnnotatedTensor._standarize_arguments(tensor=self.data, annotations=None, annotated_axis=None)
        self.assertEqual(annotations, [[], []])

    def test_check_annotations(self):
        annotations = AnnotatedTensor._check_annotations(self.data, self.annotations, 1)
        self.assertEqual(annotations, [None, self.annotations])

        annotations = AnnotatedTensor._check_annotations(self.data, None)
        self.assertEqual(annotations, [None, None])

    def test_creation(self):
        tensor = AnnotatedTensor(self.data, self.annotations, annotated_axis=1)
        self.assertEqual(tensor.shape, self.data.shape)
        self.assertEqual(tensor.annotations, [None, self.annotations])

    def test_assign_annotations(self):
        tensor = AnnotatedTensor(self.data, self.annotations, annotated_axis=1)
        new_annotations = [["new_a", "new_b", "new_c", "new_d", "new_e"], ["new_f", "new_g", "new_h", "new_i"]]
        tensor.assign_annotations(new_annotations, [0, 1])
        self.assertEqual(tensor.annotations, new_annotations)

    def test_update_annotations(self):
        tensor = AnnotatedTensor(self.data, self.annotations, annotated_axis=1)
        new_annotations = ["new_a", "new_b", "new_c", "new_d"]
        tensor.update_annotations(new_annotations, 1)
        self.assertEqual(tensor.annotations, [None, new_annotations])

    def test_annotation_axis(self):
        tensor = AnnotatedTensor(self.data, self.annotations, annotated_axis=1)
        self.assertEqual(tensor.annotated_axis(), [1])

    def test_extract_by_annotations(self):
        tensor = AnnotatedTensor(self.data, self.annotations, annotated_axis=1)
        target_annotations = ["annotation_a", "annotation_c"]
        extracted_tensor = tensor.extract_by_annotations(target_annotations, 1)
        self.assertEqual(extracted_tensor.shape, (5, 2))
        self.assertEqual(extracted_tensor.annotations, [None, ["annotation_a", "annotation_c"]])

        tensor = AnnotatedTensor(self.data, self.annotations, annotated_axis=1)
        target_annotations = [1, 3]
        extracted_tensor = tensor.extract_by_annotations(target_annotations, 1)
        self.assertEqual(extracted_tensor.shape, (5, 2))
        self.assertEqual(extracted_tensor.annotations, [None, ["annotation_b", "annotation_d"]])

    def test_new_empty(self):
        tensor = AnnotatedTensor(self.data, self.annotations, annotated_axis=1)
        empty_tensor = tensor.new_empty(5, 4)
        self.assertEqual(empty_tensor.shape, (5, 4))
        self.assertEqual(empty_tensor.annotations, [None, self.annotations])

    def test_view(self):
        tensor = AnnotatedTensor(self.data, self.annotations, annotated_axis=1)
        view_tensor = tensor.view(10, 2, annotations=["new_a", "new_b"], annotated_axis=1)
        self.assertEqual(view_tensor.shape, (10, 2))
        self.assertEqual(view_tensor.annotations, [None, ["new_a", "new_b"]])


if __name__ == '__main__':
    unittest.main()
