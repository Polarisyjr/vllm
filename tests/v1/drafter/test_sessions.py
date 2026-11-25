# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.drafter.sessions import OVERFLOW_ROW, SessionStore

BLOCK_SIZE = 16


class _BlockTables:
    """What SessionStore uses of BlockTables."""

    def __init__(self, max_num_reqs: int):
        self.block_sizes = [BLOCK_SIZE]
        self.max_num_reqs = max_num_reqs
        self.rows: dict[int, list[int]] = {}

    def append_block_ids(self, row, new_block_ids, overwrite):
        if overwrite:
            self.rows[row] = []
        self.rows[row].extend(new_block_ids[0])

    def apply_staged_writes(self):
        pass


def make_store(max_sessions=2, num_blocks=13, max_blocks_per_session=4, headroom=5):
    tables = _BlockTables(max_sessions + 1)
    return SessionStore(tables, num_blocks, max_blocks_per_session, headroom), tables


def test_sessions_take_and_return_rows_and_blocks():
    store, tables = make_store()
    # Block 0 is the null block, and the overflow row holds a full session.
    free = store.num_free_blocks
    assert free == 13 - 1 - 4

    row = store.open()
    assert row != OVERFLOW_ROW and store.num_live == 1
    # 20 tokens and 5 of headroom: 2 blocks.
    assert store.reserve(row, 20) == row
    assert len(tables.rows[row]) == 2 and store.num_free_blocks == free - 2
    assert store.reserve(row, 27) == row
    assert len(tables.rows[row]) == 2
    assert store.reserve(row, 28) == row
    assert len(tables.rows[row]) == 3
    # Never more than a full session.
    assert store.reserve(row, 10_000) == row
    assert len(tables.rows[row]) == 4

    store.close(row)
    assert store.num_live == 0 and store.num_free_blocks == free
    # The next session of the row starts its block table over.
    assert store.open() == row
    store.reserve(row, 1)
    assert len(tables.rows[row]) == 1
    assert store.num_overflowed == 0


def test_no_free_row_overflows():
    store, _ = make_store(max_sessions=2)
    assert OVERFLOW_ROW not in (store.open(), store.open())
    row = store.open()
    assert row == OVERFLOW_ROW and store.num_overflowed == 1
    assert store.reserve(row, 100) == OVERFLOW_ROW
    store.close(row)
    assert store.num_live == 2


def test_no_free_block_overflows_and_frees_the_session():
    # 6 blocks beside the null block and the overflow row's 4.
    store, _ = make_store(max_sessions=3, num_blocks=11)
    a, b = store.open(), store.open()
    assert store.reserve(a, 4 * BLOCK_SIZE - 5) == a
    assert store.reserve(b, 2 * BLOCK_SIZE - 5) == b
    assert store.num_free_blocks == 0

    assert store.reserve(b, 3 * BLOCK_SIZE - 5) == OVERFLOW_ROW
    assert store.num_overflowed == 1
    # The session's row and blocks are free for others.
    assert store.num_live == 1 and store.num_free_blocks == 2
    assert store.reserve(a, 1) == a
