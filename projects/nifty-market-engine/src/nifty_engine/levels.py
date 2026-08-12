from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .math_utils import clamp
from .models import EventKind, LevelKind, LevelMetrics, SupportResistanceLevel
from .params import StrategyParams


@dataclass(slots=True)
class _TrackerState:
    touched_at: datetime | None = None
    outside_since: datetime | None = None


class LevelClassifier:
    def __init__(self, params: StrategyParams) -> None:
        self.params = params
        self._state: dict[str, _TrackerState] = {}

    def _distance_bps(self, spot: float, level: SupportResistanceLevel) -> float:
        return (spot / level.price - 1.0) * 10_000.0

    def _breakout_direction(self, level: SupportResistanceLevel) -> float:
        return 1.0 if level.kind is LevelKind.RESISTANCE else -1.0

    def nearest_level(
        self, spot: float, levels: tuple[SupportResistanceLevel, ...]
    ) -> SupportResistanceLevel | None:
        enabled = [level for level in levels if level.enabled and level.price > 0]
        if not enabled:
            return None
        closest = min(enabled, key=lambda level: abs(self._distance_bps(spot, level)))
        if abs(self._distance_bps(spot, closest)) > self.params.level_watch_distance_bps:
            return None
        return closest

    def classify(
        self,
        *,
        now: datetime,
        spot: float,
        previous_spot: float,
        level: SupportResistanceLevel | None,
        combined_score: float,
        previous_combined_score: float,
        participation: float,
        signed_volume_acceleration: float,
    ) -> LevelMetrics:
        if level is None:
            return LevelMetrics(EventKind.NO_LEVEL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)

        state = self._state.setdefault(level.name, _TrackerState())
        distance_bps = self._distance_bps(spot, level)
        breakout_dir = self._breakout_direction(level)
        signed_distance = breakout_dir * distance_bps

        if abs(distance_bps) <= self.params.level_touch_tolerance_bps:
            state.touched_at = now

        if signed_distance > 0:
            if state.outside_since is None:
                state.outside_since = now
        else:
            state.outside_since = None

        persistence_seconds = (
            max((now - state.outside_since).total_seconds(), 0.0)
            if state.outside_since is not None else 0.0
        )
        persistence = clamp(
            persistence_seconds / max(self.params.persistence_target_seconds, 1e-6), 0.0, 1.0
        )
        penetration = clamp(
            max(signed_distance, 0.0) / max(self.params.breakout_penetration_bps, 1e-6), 0.0, 1.0
        )
        rejection = clamp(
            max(-signed_distance, 0.0) / max(self.params.rejection_depth_bps, 1e-6), 0.0, 1.0
        )

        touched_recently = False
        if state.touched_at is not None:
            touched_recently = (
                (now - state.touched_at).total_seconds() <= self.params.level_touch_memory_seconds
            )
        if not touched_recently and penetration <= 0.0:
            return LevelMetrics(
                EventKind.UNCERTAIN, 0.0, 0.0, 0.0, penetration, rejection,
                persistence, distance_bps, level.name
            )

        aligned_direction = clamp(breakout_dir * combined_score, 0.0, 1.0)
        reversal_direction = clamp(-breakout_dir * combined_score, 0.0, 1.0)
        score_change = combined_score - previous_combined_score
        breakout_accel = clamp(breakout_dir * score_change, 0.0, 1.0)
        reversal_accel = clamp(-breakout_dir * score_change, 0.0, 1.0)
        breakout_volume_accel = clamp(breakout_dir * signed_volume_acceleration, 0.0, 1.0)
        reversal_volume_accel = clamp(-breakout_dir * signed_volume_acceleration, 0.0, 1.0)

        breakout_score = clamp(
            self.params.level_direction_weight * aligned_direction
            + self.params.level_distance_weight * penetration
            + self.params.level_persistence_weight * persistence
            + self.params.level_participation_weight * participation
            + self.params.level_acceleration_weight * max(breakout_accel, breakout_volume_accel),
            0.0, 1.0,
        )
        reversal_score = clamp(
            self.params.level_direction_weight * reversal_direction
            + self.params.level_distance_weight * rejection
            + self.params.level_persistence_weight * (1.0 - persistence)
            + self.params.level_participation_weight * participation
            + self.params.level_acceleration_weight * max(reversal_accel, reversal_volume_accel),
            0.0, 1.0,
        )

        event = EventKind.UNCERTAIN
        event_score = max(breakout_score, reversal_score)
        if (
            breakout_score >= self.params.breakout_threshold
            and breakout_score - reversal_score >= self.params.decision_margin
        ):
            event = EventKind.BREAKOUT
            event_score = breakout_score
        elif (
            reversal_score >= self.params.reversal_threshold
            and reversal_score - breakout_score >= self.params.decision_margin
        ):
            event = EventKind.REVERSAL
            event_score = reversal_score

        return LevelMetrics(
            event, event_score, breakout_score, reversal_score, penetration,
            rejection, persistence, distance_bps, level.name
        )
