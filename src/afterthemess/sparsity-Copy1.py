# sparsity.py

import torch
import numpy as np

class SparsityController:
    def __init__(self, image_size, mode='random_epoch', pattern='random',
                 sparsity=0.2, block_size=5, num_blocks = 5, seed=42):
        """
        A central controller for generating conditioning and target masks
        to simulate sparse observations for diffusion training.

        Args:
            image_size (int): assumes square images of size (H, W)
            mode (str): one of ['random_epoch', 'fixed_instance', 'fixed_all']
            pattern (str): one of ['random', 'block']
            sparsity (float): total fraction of pixels to use (e.g., 0.2 = 20%)
            block_size (int): side length of square blocks (used in block mode)
            seed (int): random seed for reproducibility
        """
        self.mode = mode
        self.pattern = pattern
        self.sparsity = sparsity
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.H = self.W = image_size
        self.rng = np.random.default_rng(seed)
        self.cache = {}  # stores masks for fixed modes

    def get_masks(self, B, C, sample_ids=None):
        """
        Returns B conditioning and target masks (each of shape (C, H, W)).

        Args:
            B (int): batch size
            C (int): number of channels
            sample_ids (list or None): required for fixed_instance mode

        Returns:
            cond_masks (B x C x H x W), target_masks (B x C x H x W)
        """
        if self.mode == 'fixed_all':
            mask_cond, mask_target = self._generate_mask_pair()
            cond_masks = [mask_cond.clone() for _ in range(B)]
            target_masks = [mask_target.clone() for _ in range(B)]
            return cond_masks, target_masks

        conds, targets = [], []
        for i in range(B):
            key = sample_ids[i] if self.mode == 'fixed_instance' else self.rng.integers(0, 1e6)
            if self.mode == 'fixed_instance' and key in self.cache:
                cond, target = self.cache[key]
            else:
                cond, target = self._generate_mask_pair()
                if self.mode == 'fixed_instance':
                    self.cache[key] = (cond, target)
            conds.append(cond.repeat(C, 1, 1))
            targets.append(target.repeat(C, 1, 1))
        return conds, targets

    def _generate_mask_pair(self):
        """
        Generate a pair of masks: one for conditioning, one for supervision.

        Returns:
            cond_mask (1 x H x W), target_mask (1 x H x W)
        """
        total = self.H * self.W
        num_total = int(self.sparsity * total)
        num_half = num_total // 2

        if self.pattern == 'random':
            # Random pixel indices
            idx = self.rng.choice(total, num_total, replace=False)
            idx_cond, idx_target = idx[:num_half], idx[num_half:]
            mask_cond = torch.zeros(total)
            mask_target = torch.zeros(total)
            mask_cond[idx_cond] = 1.0
            mask_target[idx_target] = 1.0
            mask_cond = mask_cond.view(1, self.H, self.W)
            mask_target = mask_target.view(1, self.H, self.W)

        elif self.pattern == 'block':
            mask_cond = torch.zeros(self.H, self.W)
            mask_target = torch.zeros(self.H, self.W)

            total_blocks = self.num_blocks
            assert total_blocks % 2 == 0, "num_blocks must be even to split into cond/target blocks"

            block_coords = []
            max_tries = 1000
            placed = 0
            tries = 0

            while placed < total_blocks and tries < max_tries:
                x = self.rng.integers(0, self.H - self.block_size + 1)
                y = self.rng.integers(0, self.W - self.block_size + 1)

                # Check for overlap
                overlap = False
                for bx, by in block_coords:
                    if abs(bx - x) < self.block_size and abs(by - y) < self.block_size:
                        overlap = True
                        break

                if not overlap:
                    block_coords.append((x, y))
                    placed += 1
                tries += 1

            assert len(block_coords) == total_blocks, f"Failed to place {total_blocks} blocks"

            # Shuffle and split into two equal halves
            self.rng.shuffle(block_coords)
            cond_blocks = block_coords[:total_blocks // 2]
            target_blocks = block_coords[total_blocks // 2:]

            for x, y in cond_blocks:
                mask_cond[x:x+self.block_size, y:y+self.block_size] = 1

            for x, y in target_blocks:
                mask_target[x:x+self.block_size, y:y+self.block_size] = 1

            mask_cond = mask_cond.view(1, self.H, self.W)
            mask_target = mask_target.view(1, self.H, self.W)


        else:
            raise ValueError(f"Unknown pattern type: {self.pattern}")

        return mask_cond, mask_target
