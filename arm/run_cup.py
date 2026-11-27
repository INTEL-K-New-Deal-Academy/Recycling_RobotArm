"""컵 분리 동작을 실행한다. Ctrl+C 로 멈추면 대기 자세 → 토크 끄기 순서로 안전하게 멈춘다.

    python run_cup.py                    # config.yaml 의 port 로 컵 1개 처리
    python run_cup.py --port fake        # 가상 팔로 흐름만 확인
    python run_cup.py --port COM4 --slow # 처음 시험할 때: 모든 동작을 느리게
    python run_cup.py --step pour        # 한 단계만 (home, pick, lid, pour, drop)
    python run_cup.py --count 3          # 컵 3개 연속 (Enter 를 눌러 다음 컵)
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys

from cup_arm import CupArm

STEPS = {
    "home": CupArm.home,
    "pick": CupArm.pick_cup,
    "lid": CupArm.remove_lid,
    "pour": CupArm.pour,
    "drop": CupArm.drop_cup,
    "all": CupArm.process_cup,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="컵 분리 로봇팔 실행")
    ap.add_argument("--port", help='포트 (기본: config.yaml). "fake" = 가상 팔')
    ap.add_argument("--slow", action="store_true", help="처음 시험하는 동작: 모두 느리게")
    ap.add_argument("--step", choices=STEPS, default="all", help="실행할 단계 (기본: all)")
    ap.add_argument("--count", type=int, default=1, help="반복 횟수 (기본 1)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("arm")

    # docker stop 등 SIGTERM 도 Ctrl+C 와 같게 처리
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    arm = CupArm.from_config(args.port, slow=args.slow)
    missing = arm.missing_poses()
    if missing and not arm.is_fake:
        log.error("아직 저장하지 않은 자세가 있습니다: %s  →  python teach.py 로 먼저 저장하세요", missing)
        arm.bot.torque(False)
        arm.close()
        return 1

    try:
        for i in range(args.count):
            if i > 0:
                input(f"[{i + 1}/{args.count}] 다음 컵을 놓고 Enter ")
            log.info("=== %s 시작 (%d/%d) ===", args.step, i + 1, args.count)
            STEPS[args.step](arm)
            log.info("=== %s 완료 ===", args.step)
    except KeyboardInterrupt:
        log.warning("Ctrl+C: 대기 자세로 돌아간 뒤 토크를 끕니다")
    finally:
        arm.safe_stop()
        arm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
