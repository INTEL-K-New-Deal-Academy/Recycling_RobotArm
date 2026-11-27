"""컵 하나를 분리수거하는 로봇팔 동작 모음.

동작 순서:
    대기 → 컵 중간 잡고 들기 → 고정 고리에 뚜껑 가장자리 걸기 → 내리며 기울여 뚜껑·빨대 벗기기
    → 배수통에 음료·얼음 붓기 → 종이류함에 빈 컵 버리기 → 대기

(2026-11-27 실물 시험에서 「컵 윗부분을 눌러 뚜껑 헐겁게」 방식이 실패해 고리 방식으로 바꿨다)

단계마다 함수를 따로 두어서, 메인 프로그램(C 담당)이 필요한 단계만 부르거나
나중에 팝콘 통처럼 다른 부품 처리를 끼워 넣기 쉽게 했다.

    from cup_arm import CupArm

    with CupArm.from_config() as arm:     # config.yaml 의 port 사용 ("fake" = 가상 팔)
        arm.process_cup()                 # 컵 하나 전체 처리
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import yaml

from soarm import Robot, RobotError

log = logging.getLogger("arm")

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class CupArm:
    """SO-ARM101 로 컵 하나를 분리하는 동작 묶음. 실제 팔과 가상 팔(port="fake")에서 같은 코드가 돈다."""

    def __init__(self, robot: Robot, config: dict, *, slow: bool = False):
        self.bot = robot
        self.cfg = config
        sp = config["speed"]
        # --slow: 처음 시험하는 동작은 모두 느리게 (CLAUDE.md 4.3)
        self.speed_normal = sp["test"] if slow else sp["normal"]
        self.speed_carry = sp["test"] if slow else sp["slow"]
        self.speed_hook = min(sp["test"], sp["hook"]) if slow else sp["hook"]
        self.timeout = float(config.get("pose_timeout_sec", 10.0))

    @classmethod
    def from_config(cls, port: Optional[str] = None, *, slow: bool = False,
                    config_path: Path = CONFIG_PATH) -> "CupArm":
        cfg = load_config(config_path)
        robot = Robot(port or cfg["port"], speed=cfg["speed"]["normal"])
        arm = cls(robot, cfg, slow=slow)
        robot.torque(True)            # 지금 자세 그대로 잠근다 (튀지 않음)
        return arm

    # ── 기본 도구 ────────────────────────────────────────────────────────
    @property
    def is_fake(self) -> bool:
        return not self.bot.real

    def missing_poses(self) -> list[str]:
        """config.yaml 에 적힌 자세 중 아직 teach.py 로 저장하지 않은 것"""
        saved = self.bot.poses()
        return [name for name in self.cfg["poses"].values() if name not in saved]

    def go(self, key: str, speed: Optional[int] = None) -> None:
        """config.yaml poses 의 key 에 해당하는 자세로 이동 (그리퍼는 그대로)"""
        name = self.cfg["poses"][key]
        log.info("자세 이동: %s", name)
        try:
            self.bot.go_pose(name, speed=speed or self.speed_normal, timeout=self.timeout)
        except RobotError:
            if self.is_fake and name not in self.bot.poses():
                # 가상 팔: 아직 저장 안 한 자세는 건너뛰고 흐름만 확인한다
                log.warning("[가상 팔] 자세 '%s' 가 아직 없어 건너뜁니다", name)
                return
            raise

    def grip(self, key: str) -> None:
        """config.yaml gripper 의 key 값(%)으로 그리퍼 열기·닫기"""
        pct = float(self.cfg["gripper"][key])
        log.info("그리퍼 %s: %.0f%%", key, pct)
        self.bot.gripper(pct)
        self.bot.wait_until_settled(timeout=3.0)
        time.sleep(self.cfg["wait"]["grip_settle_sec"])

    # ── 분리 단계 ────────────────────────────────────────────────────────
    def home(self) -> None:
        self.go("home")

    def pick_cup(self) -> None:
        """컵에 다가가 중간~아래를 잡고 들어 올린다 (위쪽은 고리가 들어갈 자리로 비워 둔다)"""
        self.grip("open")
        self.go("approach_cup")
        self.go("grip_cup")
        self.grip("hold")
        self.go("lift_cup", speed=self.speed_carry)

    def remove_lid(self) -> None:
        """고정 고리에 뚜껑 가장자리를 걸고, 컵을 내리며 기울여 뚜껑·빨대를 벗겨 플라스틱함으로 떨어뜨린다.

        병따개처럼 한쪽부터 들어 올리므로 적은 힘으로 벗겨진다.
        걸고 당기는 구간은 느리게 — 빠르면 뚜껑이 튀고 음료가 출렁인다.
        """
        self.go("hook_ready", speed=self.speed_carry)
        self.go("hook_engage", speed=self.speed_hook)
        self.go("hook_pull", speed=self.speed_hook)
        self.go("hook_clear", speed=self.speed_hook)
        time.sleep(self.cfg["wait"]["lid_drop_sec"])

    def pour(self) -> None:
        """배수통 위에서 기울여 음료와 얼음을 한꺼번에 버린다"""
        self.go("pour_ready", speed=self.speed_carry)
        self.go("pour_tilt", speed=self.speed_carry)
        time.sleep(self.cfg["wait"]["pour_sec"])
        self.go("pour_ready")

    def drop_cup(self) -> None:
        """종이류함 위에서 빈 컵을 놓는다"""
        self.go("paper_bin")
        self.grip("release")

    def process_cup(self) -> None:
        """컵 하나 전체 처리. 끝나면 대기 자세"""
        self.pick_cup()
        self.remove_lid()
        self.pour()
        self.drop_cup()
        self.home()

    # ── 안전 정지 ────────────────────────────────────────────────────────
    def safe_stop(self) -> None:
        """대기 자세로 돌아간 뒤 토크를 끈다 (CLAUDE.md 4.3). 돌아가다 실패해도 토크는 끈다"""
        try:
            self.bot.stop()
            self.go("home")
        except Exception as e:     # 정지 중에는 어떤 오류여도 토크 끄기까지 간다
            log.error("대기 자세 복귀 실패: %s", e)
        finally:
            self.bot.torque(False)
            log.info("토크 끔")

    def close(self) -> None:
        self.bot.close()

    def __enter__(self) -> "CupArm":
        return self

    def __exit__(self, *_exc) -> None:
        self.safe_stop()
        self.close()
