# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.block_table import BlockTables

logger = init_logger(__name__)

# Sessions the engine has no row or no KV blocks for share this row: a safety
# net, not a mode. What they write stays in the row's own blocks; their drafts
# are noise, which the target's verification rejects.
OVERFLOW_ROW = 0


class SessionStore:
    """Rows and KV blocks of the sessions the engine keeps state for.

    A session is a row: of the block table, and of every per-request buffer
    the speculators index. The store hands out rows and grows a row's KV
    blocks as its session gets longer; which client a row belongs to is the
    engine's business.
    """

    def __init__(
        self,
        block_tables: BlockTables,
        num_blocks: int,
        max_blocks_per_session: int,
        headroom: int,
    ):
        self.block_tables = block_tables
        # Tokens a session writes past its length: the drafter's own draft
        # positions.
        self.headroom = headroom
        self.block_size = block_tables.block_sizes[0]
        self.num_blocks = num_blocks
        self.max_blocks_per_session = max_blocks_per_session
        # A row per session, and the overflow row.
        self.max_sessions = block_tables.max_num_reqs - 1
        self._free_rows = list(range(self.max_sessions, OVERFLOW_ROW, -1))
        self._free_blocks = list(range(num_blocks - 1, 0, -1))  # block 0 stays null
        self._blocks: dict[int, list[int]] = {}
        self.num_overflowed = 0

        overflow = self._take_blocks(max_blocks_per_session)
        if overflow is None:
            raise RuntimeError("the drafter KV cache is smaller than one session")
        block_tables.append_block_ids(OVERFLOW_ROW, (overflow,), overwrite=True)
        block_tables.apply_staged_writes()

    @property
    def num_live(self) -> int:
        return self.max_sessions - len(self._free_rows)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def open(self) -> int:
        """Row for a new session, or OVERFLOW_ROW when every row is taken."""
        if not self._free_rows:
            self._overflow("row")
            return OVERFLOW_ROW
        return self._free_rows.pop()

    def reserve(self, row: int, seq_len: int) -> int:
        """Make sure `row` has KV blocks for a session of `seq_len` tokens.

        Returns the row. When the cache is full the session loses its row and
        OVERFLOW_ROW is returned. Block table writes are staged; the runner
        applies them at the next round.
        """
        if row == OVERFLOW_ROW:
            return row
        need = min(
            cdiv(seq_len + self.headroom, self.block_size),
            self.max_blocks_per_session,
        )
        have = self._blocks.setdefault(row, [])
        if need > len(have):
            new = self._take_blocks(need - len(have))
            if new is None:
                self.close(row)
                self._overflow("KV block")
                return OVERFLOW_ROW
            # A row's first blocks overwrite the count its last session left.
            self.block_tables.append_block_ids(row, (new,), overwrite=not have)
            have.extend(new)
        return row

    def close(self, row: int) -> None:
        if row == OVERFLOW_ROW:
            return
        self._free_blocks.extend(reversed(self._blocks.pop(row, ())))
        self._free_rows.append(row)

    def _take_blocks(self, n: int) -> list[int] | None:
        if n > len(self._free_blocks):
            return None
        blocks = self._free_blocks[-n:][::-1]
        del self._free_blocks[-n:]
        return blocks

    def _overflow(self, what: str) -> None:
        self.num_overflowed += 1
        logger.error_once(
            "The drafter has no free %s for a session; its drafts are unusable. "
            "Raise --max-live-sessions or route fewer targets here.",
            what,
        )
