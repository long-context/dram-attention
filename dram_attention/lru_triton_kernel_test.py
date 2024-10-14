import unittest

import torch

from dram_attention.lru_triton_kernel import load_lru_cache_


class TestLRUCacheLoad(unittest.TestCase):
    def setUp(self):
        self.device = "cuda"
        self.num_heads = 2
        self.num_dram_pages = 8
        self.num_hbm_pages = 4
        self.page_dim = 64
        self.k = 2  # Number of top-k pages to load

    def test_lru_cache_load_basic(self):
        # Initialize test data
        dram_kv_cache = torch.randn(
            self.num_heads, self.num_dram_pages, self.page_dim, device=self.device
        )
        hbm_kv_cache = torch.zeros(
            self.num_heads, self.num_hbm_pages, self.page_dim, device=self.device
        )
        page_access_time = torch.arange(
            self.num_hbm_pages, dtype=torch.int32, device=self.device
        ).repeat(self.num_heads, 1)
        dram_page_to_hbm_page_mapping = torch.full(
            (self.num_heads, self.num_dram_pages),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        hbm_page_to_dram_page_mapping = torch.full(
            (self.num_heads, self.num_hbm_pages),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        topk_dram_page_index = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.int32, device=self.device
        )
        current_step = torch.tensor(
            [self.num_hbm_pages], dtype=torch.int32, device=self.device
        )

        # Run the function
        load_lru_cache_(
            dram_kv_cache=dram_kv_cache,
            page_access_time=page_access_time,
            dram_page_to_hbm_page_mapping=dram_page_to_hbm_page_mapping,
            hbm_page_to_dram_page_mapping=hbm_page_to_dram_page_mapping,
            topk_dram_page_index=topk_dram_page_index,
            current_step=current_step,
            hbm_kv_cache=hbm_kv_cache,
        )

        # Check that the correct pages were loaded
        self.assertTrue(torch.allclose(hbm_kv_cache[0, 0], dram_kv_cache[0, 0]))
        self.assertTrue(torch.allclose(hbm_kv_cache[0, 1], dram_kv_cache[0, 1]))
        self.assertTrue(torch.allclose(hbm_kv_cache[1, 0], dram_kv_cache[1, 1]))
        self.assertTrue(torch.allclose(hbm_kv_cache[1, 1], dram_kv_cache[1, 2]))

        # Check that the metadata was updated correctly
        self.assertEqual(page_access_time[0, 0].item(), self.num_hbm_pages)
        self.assertEqual(page_access_time[0, 1].item(), self.num_hbm_pages)
        self.assertEqual(page_access_time[1, 0].item(), self.num_hbm_pages)
        self.assertEqual(page_access_time[1, 1].item(), self.num_hbm_pages)
        self.assertEqual(dram_page_to_hbm_page_mapping[0, 0].item(), 0)
        self.assertEqual(dram_page_to_hbm_page_mapping[0, 1].item(), 1)
        self.assertEqual(dram_page_to_hbm_page_mapping[1, 1].item(), 0)
        self.assertEqual(dram_page_to_hbm_page_mapping[1, 2].item(), 1)
        self.assertEqual(hbm_page_to_dram_page_mapping[0, 0].item(), 0)
        self.assertEqual(hbm_page_to_dram_page_mapping[0, 1].item(), 1)
        self.assertEqual(hbm_page_to_dram_page_mapping[1, 0].item(), 1)
        self.assertEqual(hbm_page_to_dram_page_mapping[1, 1].item(), 2)

    def test_lru_cache_load_eviction(self):
        # Initialize test data
        dram_kv_cache = torch.randn(
            self.num_heads, self.num_dram_pages, self.page_dim, device=self.device
        )
        hbm_kv_cache = torch.zeros(
            self.num_heads, self.num_hbm_pages, self.page_dim, device=self.device
        )
        page_access_time = torch.arange(
            self.num_hbm_pages, dtype=torch.int32, device=self.device
        ).repeat(self.num_heads, 1)
        dram_page_to_hbm_page_mapping = torch.full(
            (self.num_heads, self.num_dram_pages),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        hbm_page_to_dram_page_mapping = torch.full(
            (self.num_heads, self.num_hbm_pages),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        topk_dram_page_index = torch.tensor(
            [[4, 5], [5, 6]], dtype=torch.int32, device=self.device
        )
        current_step = torch.tensor(
            [self.num_hbm_pages], dtype=torch.int32, device=self.device
        )

        # Fill HBM cache
        for i in range(self.num_hbm_pages):
            hbm_kv_cache[:, i] = dram_kv_cache[:, i]
            dram_page_to_hbm_page_mapping[:, i] = i
            hbm_page_to_dram_page_mapping[:, i] = i

        # Run the function
        load_lru_cache_(
            dram_kv_cache=dram_kv_cache,
            page_access_time=page_access_time,
            dram_page_to_hbm_page_mapping=dram_page_to_hbm_page_mapping,
            hbm_page_to_dram_page_mapping=hbm_page_to_dram_page_mapping,
            topk_dram_page_index=topk_dram_page_index,
            current_step=current_step,
            hbm_kv_cache=hbm_kv_cache,
        )

        # Check that the new pages were loaded and the least recently used pages were evicted
        self.assertTrue(torch.allclose(hbm_kv_cache[0, 0], dram_kv_cache[0, 4]))
        self.assertTrue(torch.allclose(hbm_kv_cache[0, 1], dram_kv_cache[0, 5]))
        self.assertTrue(torch.allclose(hbm_kv_cache[1, 0], dram_kv_cache[1, 5]))
        self.assertTrue(torch.allclose(hbm_kv_cache[1, 1], dram_kv_cache[1, 6]))

        # Check that the metadata was updated correctly
        self.assertEqual(dram_page_to_hbm_page_mapping[0, 4].item(), 0)
        self.assertEqual(dram_page_to_hbm_page_mapping[0, 5].item(), 1)
        self.assertEqual(dram_page_to_hbm_page_mapping[1, 5].item(), 0)
        self.assertEqual(dram_page_to_hbm_page_mapping[1, 6].item(), 1)
        self.assertEqual(hbm_page_to_dram_page_mapping[0, 0].item(), 4)
        self.assertEqual(hbm_page_to_dram_page_mapping[0, 1].item(), 5)
        self.assertEqual(hbm_page_to_dram_page_mapping[1, 0].item(), 5)
        self.assertEqual(hbm_page_to_dram_page_mapping[1, 1].item(), 6)
        self.assertEqual(page_access_time[0, 0].item(), self.num_hbm_pages)
        self.assertEqual(page_access_time[0, 1].item(), self.num_hbm_pages)
        self.assertEqual(page_access_time[1, 0].item(), self.num_hbm_pages)
        self.assertEqual(page_access_time[1, 1].item(), self.num_hbm_pages)

    def test_lru_cache_load_multiple_heads(self):
        # Initialize test data
        dram_kv_cache = torch.randn(
            self.num_heads, self.num_dram_pages, self.page_dim, device=self.device
        )
        hbm_kv_cache = torch.zeros(
            self.num_heads, self.num_hbm_pages, self.page_dim, device=self.device
        )
        page_access_time = torch.arange(
            self.num_hbm_pages, dtype=torch.int32, device=self.device
        ).repeat(self.num_heads, 1)
        dram_page_to_hbm_page_mapping = torch.full(
            (self.num_heads, self.num_dram_pages),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        hbm_page_to_dram_page_mapping = torch.full(
            (self.num_heads, self.num_hbm_pages),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        topk_dram_page_index = torch.tensor(
            [[0, 2], [1, 3]], dtype=torch.int32, device=self.device
        )
        current_step = torch.tensor(
            [self.num_hbm_pages], dtype=torch.int32, device=self.device
        )

        # Run the function
        load_lru_cache_(
            dram_kv_cache=dram_kv_cache,
            page_access_time=page_access_time,
            dram_page_to_hbm_page_mapping=dram_page_to_hbm_page_mapping,
            hbm_page_to_dram_page_mapping=hbm_page_to_dram_page_mapping,
            topk_dram_page_index=topk_dram_page_index,
            current_step=current_step,
            hbm_kv_cache=hbm_kv_cache,
        )

        # Check that the correct pages were loaded for each head
        self.assertTrue(torch.allclose(hbm_kv_cache[0, 0], dram_kv_cache[0, 0]))
        self.assertTrue(torch.allclose(hbm_kv_cache[0, 1], dram_kv_cache[0, 2]))
        self.assertTrue(torch.allclose(hbm_kv_cache[1, 0], dram_kv_cache[1, 1]))
        self.assertTrue(torch.allclose(hbm_kv_cache[1, 1], dram_kv_cache[1, 3]))

        # Check that the metadata was updated correctly for each head
        self.assertEqual(page_access_time[0, 0].item(), self.num_hbm_pages)
        self.assertEqual(page_access_time[0, 1].item(), self.num_hbm_pages)
        self.assertEqual(page_access_time[1, 0].item(), self.num_hbm_pages)
        self.assertEqual(page_access_time[1, 1].item(), self.num_hbm_pages)
        self.assertEqual(dram_page_to_hbm_page_mapping[0, 0].item(), 0)
        self.assertEqual(dram_page_to_hbm_page_mapping[0, 2].item(), 1)
        self.assertEqual(dram_page_to_hbm_page_mapping[1, 1].item(), 0)
        self.assertEqual(dram_page_to_hbm_page_mapping[1, 3].item(), 1)
        self.assertEqual(hbm_page_to_dram_page_mapping[0, 0].item(), 0)
        self.assertEqual(hbm_page_to_dram_page_mapping[0, 1].item(), 2)
        self.assertEqual(hbm_page_to_dram_page_mapping[1, 0].item(), 1)
        self.assertEqual(hbm_page_to_dram_page_mapping[1, 1].item(), 3)


if __name__ == "__main__":
    unittest.main()
