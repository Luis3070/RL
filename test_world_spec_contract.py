from __future__ import annotations

import numpy as np
import pytest

from pybh_backend import (
    WorldPlayerSpec,
    WorldSpec,
    build_world_spec,
    simrobot_body_name_for_player,
    validate_world_spec,
)


def test_build_world_spec_coerces_players_and_preserves_lists():
    world = build_world_spec(
        ball_pos=np.array([100.0, 200.0], dtype=np.float32),
        robot_pos=np.array([-300.0, 50.0], dtype=np.float32),
        robot_theta=0.25,
        teammates=[(2, 400.0, 0.0, 0.5)],
        opponents=[{"number": 3, "x": 700.0, "y": -200.0, "theta": -0.3, "upright": True}],
    )

    assert isinstance(world, WorldSpec)
    assert world.ball_x == 100.0
    assert world.robot_x == -300.0
    assert world.teammates == (WorldPlayerSpec(number=2, x=400.0, y=0.0, theta=0.5, upright=True),)
    assert world.opponents == (WorldPlayerSpec(number=3, x=700.0, y=-200.0, theta=-0.3, upright=True),)


def test_world_spec_rejects_controlled_robot_in_teammates():
    world = WorldSpec(
        ball_x=0.0,
        ball_y=0.0,
        robot_x=0.0,
        robot_y=0.0,
        robot_theta=0.0,
        teammates=(WorldPlayerSpec(number=1, x=100.0, y=0.0),),
    )

    with pytest.raises(ValueError, match="controlled robot"):
        validate_world_spec(world, mode="simrobot")


def test_world_spec_rejects_non_upright_in_visible_backends():
    world = WorldSpec(
        ball_x=0.0,
        ball_y=0.0,
        robot_x=0.0,
        robot_y=0.0,
        robot_theta=0.0,
        opponents=(WorldPlayerSpec(number=4, x=500.0, y=100.0, upright=False),),
    )

    with pytest.raises(ValueError, match="upright=False"):
        validate_world_spec(world, mode="simrobot_visible")


def test_simrobot_body_name_for_player_uses_absolute_slots():
    assert simrobot_body_name_for_player(7, opponent=False) == "robot7"
    assert simrobot_body_name_for_player(7, opponent=True) == "robot27"
