import unittest

import torch

from models.C_relora import SRHTBasisTransform


class SRHTBasisTransformTest(unittest.TestCase):
    def test_transform_is_orthonormal_and_non_trivial(self) -> None:
        dim = 8
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1234)

        basis = SRHTBasisTransform(dim, torch.device("cpu"), generator)

        identity = torch.eye(dim)
        transform = basis.right_multiply(identity)
        print("SRHT transform (dim=8):\n", transform)

        gram = transform.T @ transform
        self.assertTrue(
            torch.allclose(gram, torch.eye(dim), atol=1e-5, rtol=1e-5),
            msg="SRHT transform should be orthonormal",
        )

        self.assertFalse(
            torch.allclose(transform, identity, atol=1e-6),
            msg="SRHT transform should not be the identity matrix",
        )

    def test_transpose_round_trip_matches_input(self) -> None:
        dim = 7
        generator = torch.Generator(device="cpu")
        generator.manual_seed(9876)
        basis = SRHTBasisTransform(dim, torch.device("cpu"), generator)

        x = torch.randn(3, dim)
        forward = basis.right_multiply(x)
        back = basis.right_multiply_transpose(forward)

        self.assertTrue(
            torch.allclose(back, x, atol=1e-5, rtol=1e-5),
            msg="SRHT transpose should invert the forward transform",
        )


if __name__ == "__main__":
    unittest.main()
