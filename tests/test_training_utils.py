import unittest

import torch
from torch.optim import AdamW

from utils.training_utils import create_scheduler


class CosineRestartSchedulerTest(unittest.TestCase):
    def test_scheduler_resets_align_with_interval(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = AdamW(model.parameters(), lr=5e-4)
        scheduler = create_scheduler(
            optimizer,
            {
                "type": "cosine_restarts",
                "first_warmup_steps": 2000,
                "restart_every": 4000,
                "restart_warmup_steps": 200,
                "min_lr_ratio": 0.1,
            },
            total_steps=20000,
        )

        checkpoints = (1999, 2000, 3999, 4000, 4001, 7999, 8000)
        lrs = {}
        for step in range(0, max(checkpoints) + 1):
            optimizer.zero_grad()
            optimizer.step()
            scheduler.step()
            if step in checkpoints:
                lrs[step] = optimizer.param_groups[0]["lr"]

        self.assertAlmostEqual(lrs[3999], 0.0, delta=1e-12)
        self.assertGreater(lrs[1999], lrs[3999])
        self.assertLess(lrs[4000], lrs[2000])
        self.assertGreater(lrs[4001], lrs[4000])
        self.assertAlmostEqual(lrs[7999], 0.0, delta=1e-12)
        self.assertLess(lrs[8000], lrs[4001])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
