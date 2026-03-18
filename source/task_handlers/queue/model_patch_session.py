"""Helpers for temporary model patch sessions."""

from __future__ import annotations


class ModelPatchSession:
    @staticmethod
    def _snapshot_patch_keys(target: dict) -> dict:
        return dict(target)

    @staticmethod
    def _apply_svi_patch_values(target: dict) -> None:
        target["svi2pro"] = True
        target["sliding_window"] = True

    @staticmethod
    def _restore_snapshot(target: dict, snapshot: dict) -> None:
        target.clear()
        target.update(snapshot)
