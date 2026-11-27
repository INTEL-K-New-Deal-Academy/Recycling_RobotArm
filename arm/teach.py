"""자세 저장(티칭) 도구: 토크를 끈 팔을 손으로 옮기고 Enter 를 눌러 자세를 저장한다.

    python teach.py --port COM4              # config.yaml 의 자세를 순서대로 하나씩 저장
    python teach.py --port COM4 --only pour_tilt   # 자세 하나만 다시 저장
    python teach.py --port COM4 --list       # 저장된 자세 목록

저장 위치: data/poses.json (팀이 같은 값을 쓰도록 Git 에 올린다)
그리퍼 값은 자세와 함께 저장되지만, 동작 중에는 config.yaml gripper 값을 쓴다.
"""
from __future__ import annotations

import argparse
import sys

from cup_arm import load_config
from soarm import Robot


def main(argv=None) -> int:
    cfg = load_config()
    names = list(cfg["poses"].values())

    ap = argparse.ArgumentParser(description="자세 저장 도구")
    ap.add_argument("--port", help='포트 (기본: config.yaml). "fake" = 가상 팔')
    ap.add_argument("--only", choices=names, help="이 자세 하나만 저장")
    ap.add_argument("--list", action="store_true", help="저장된 자세 목록만 보기")
    args = ap.parse_args(argv)

    bot = Robot(args.port or cfg["port"])
    try:
        saved = bot.poses()
        if args.list:
            for n in names:
                mark = "저장됨" if n in saved else "없음  "
                when = saved.get(n, {}).get("timestamp", "")
                print(f"  [{mark}] {n:18s} {when}")
            return 0

        targets = [args.only] if args.only else names
        bot.torque(False)
        print("토크를 껐습니다. 팔을 손으로 받쳐 주세요 (힘이 빠져 내려올 수 있음).")
        for n in targets:
            note = " (이미 있음, 덮어씀)" if n in saved else ""
            ans = input(f"\n[{n}]{note} 팔을 옮기고 Enter, 건너뛰려면 s, 그만하려면 q: ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue
            bot.save_pose(n)
            print(f"  저장: {n}  {[round(a, 1) for a in bot.angles().values()]}")
    except KeyboardInterrupt:
        print("\n중단")
    finally:
        bot.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
