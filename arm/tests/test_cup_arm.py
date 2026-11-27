"""cup_arm 시험 — 가상 팔(port="fake")로, 실물 없이 동작 순서와 안전 정지를 확인한다.

    cd arm && python -m pytest tests -q
"""
import copy
import sys
from pathlib import Path

import pytest

ARM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARM_DIR))

import soarm.robot as soarm_robot  # noqa: E402
from soarm.fake import FOLDED, FakeBus, FakeSerial  # noqa: E402

import cup_arm  # noqa: E402
from cup_arm import CupArm, load_config  # noqa: E402


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["port"] = "fake"
    c["speed"] = {k: 4000 for k in c["speed"]}   # 시험은 빠르게
    c["wait"] = {k: 0 for k in c["wait"]}
    return c


@pytest.fixture
def poses_file(tmp_path, monkeypatch):
    """실제 data/poses.json 을 건드리지 않도록 임시 파일로 바꾼다"""
    p = tmp_path / "poses.json"
    monkeypatch.setattr(soarm_robot, "POSES_FILE", p)
    return p


@pytest.fixture
def make_arm(cfg, poses_file, monkeypatch):
    def _make(teach_all=True):
        FakeSerial.bus = FakeBus(present=FOLDED)       # 시험마다 새 가상 팔
        cfg_path = poses_file.parent / "config.yaml"
        monkeypatch.setattr(cup_arm, "load_config", lambda *_a, **_k: cfg)
        arm = CupArm.from_config(config_path=cfg_path)
        if teach_all:
            for i, name in enumerate(cfg["poses"].values()):
                arm.bot.move_all({"base": -10 + 2 * i, "elbow": 5 + i})   # 자세마다 다른 각도
                arm.bot.wait_until_settled(timeout=2)
                arm.bot.save_pose(name)
            arm.bot.go_pose(cfg["poses"]["home"], timeout=2)
        return arm
    return _make


def record_moves(arm, monkeypatch):
    visited = []
    real = arm.bot.go_pose

    def spy(name, *a, **k):
        visited.append(name)
        return real(name, *a, **k)

    monkeypatch.setattr(arm.bot, "go_pose", spy)
    return visited


def test_process_cup_order(make_arm, monkeypatch):
    arm = make_arm()
    visited = record_moves(arm, monkeypatch)
    arm.process_cup()
    assert visited == [
        "approach_cup", "grip_cup", "lift_cup",
        "hook_ready", "hook_engage", "hook_pull", "hook_clear",
        "pour_ready", "pour_tilt", "pour_ready",
        "paper_bin", "home",
    ]


def test_arrives_at_taught_pose(make_arm):
    arm = make_arm()
    target = arm.bot.poses()["pour_tilt"]["angles"]
    arm.go("pour_tilt")
    now = arm.bot.angles()
    for sid in range(1, 6):                     # 그리퍼(6) 제외
        assert now[sid] == pytest.approx(target[sid], abs=1.0)


def test_safe_stop_goes_home_then_torque_off(make_arm, monkeypatch):
    arm = make_arm()
    arm.go("pour_tilt")
    visited = record_moves(arm, monkeypatch)
    arm.safe_stop()
    assert visited == ["home"]
    assert not any(arm.bot.driver.get_torque(s) for s in range(1, 7))


def test_safe_stop_turns_torque_off_even_if_home_fails(make_arm, monkeypatch):
    arm = make_arm()

    def boom(*_a, **_k):
        raise RuntimeError("이동 실패")

    monkeypatch.setattr(arm.bot, "go_pose", boom)
    arm.safe_stop()
    assert not any(arm.bot.driver.get_torque(s) for s in range(1, 7))


def test_fake_skips_untaught_poses(make_arm, cfg):
    arm = make_arm(teach_all=False)
    assert set(arm.missing_poses()) == set(cfg["poses"].values())
    arm.process_cup()                           # 자세가 없어도 가상 팔은 끝까지 돈다


def test_pick_opens_then_holds(make_arm, monkeypatch):
    arm = make_arm()
    sent = []
    monkeypatch.setattr(arm.bot, "gripper", lambda pct, *a, **k: sent.append(pct))
    arm.pick_cup()
    g = arm.cfg["gripper"]
    assert sent == [g["open"], g["hold"]]       # 누르기(squeeze) 없이 잡기만


def test_hook_moves_are_slow(make_arm, monkeypatch):
    """고리에 걸고 당기는 구간은 hook 속도로 움직인다"""
    arm = make_arm()
    arm.speed_hook = 123
    speeds = {}
    real = arm.bot.go_pose

    def spy(name, *a, speed=None, **k):
        speeds[name] = speed
        return real(name, *a, speed=speed, **k)

    monkeypatch.setattr(arm.bot, "go_pose", spy)
    arm.remove_lid()
    assert [speeds[n] for n in ("hook_engage", "hook_pull", "hook_clear")] == [123] * 3


def test_slow_option_never_speeds_up_hook():
    cfg = {"speed": {"normal": 200, "slow": 80, "test": 60, "hook": 50}}
    arm = CupArm(robot=None, config=cfg, slow=True)
    assert arm.speed_hook == 50                 # --slow 여도 hook 속도보다 빨라지지 않음
