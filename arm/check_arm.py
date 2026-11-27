"""로봇팔 연결 점검.

    python check_arm.py --port COM6           # 읽기만: 모터 6개 응답·각도·전압·온도 (팔이 움직이지 않음)
    python check_arm.py --port COM6 --move    # 관절마다 약 5°씩 천천히 움직였다가 제자리로 (Enter 로 하나씩)
    python check_arm.py --port COM6 --move --yes   # 묻지 않고 차례로

--move 는 지금 자세에서 토크를 켜고(튀지 않음), 관절마다 가동범위 가운데 쪽으로 조금만 움직인다.
끝나거나 Ctrl+C 로 멈추면 토크를 끈다 — 팔이 힘이 빠지므로 접힌 자세에서 하거나 손으로 받친다.
"""
from __future__ import annotations

import argparse
import sys

from soarm import Robot
from soarm.driver_sdk import JOINT_IDS, JOINT_LIMITS, JOINT_NAMES

STEP_TICKS = {1: 57, 2: 57, 3: 57, 4: 57, 5: 57, 6: 150}   # 57틱 ≈ 5°, 그리퍼는 조금 더
SPEED = 60          # 처음 움직이는 동작이라 느리게 (config.yaml speed.test 와 같음)
OK_TICKS = 20       # 목표와 이만큼 안이면 도착으로 본다


def show_status(bot: Robot) -> bool:
    d = bot.driver
    ok = True
    print(f"\n{bot.port}")
    print(f"  {'ID':<3}{'joint':<13}{'pos':>6}{'deg':>8}{'range':>13}{'torque':>8}{'V':>7}{'C':>5}")
    for sid in JOINT_IDS:
        if not d.ping(sid):
            print(f"  {sid:<3}{JOINT_NAMES[sid - 1]:<13}응답 없음")
            ok = False
            continue
        pos, lim = d.get_position(sid), JOINT_LIMITS[sid]
        inside = lim["min"] <= pos <= lim["max"] if lim["max"] <= 4095 else True
        print(f"  {sid:<3}{JOINT_NAMES[sid - 1]:<13}{pos:>6}{d.position_to_degrees(pos):>8.1f}"
              f"{lim['min']:>6}~{lim['max']:<6}{'on' if d.get_torque(sid) else 'off':>6}"
              f"{d.get_voltage(sid):>7.1f}{d.get_temperature(sid):>5}{'' if inside else '  ← 범위 밖'}")
    return ok


def wiggle(bot: Robot, sid: int) -> bool:
    lim = JOINT_LIMITS[sid]
    start = bot.driver.get_position(sid)
    mid = (lim["min"] + lim["max"]) / 2
    step = STEP_TICKS[sid] if start < mid else -STEP_TICKS[sid]     # 가동범위 가운데 쪽으로
    results = []
    for goal in (start + step, start):
        bot.move_raw(sid, goal, speed=SPEED)
        bot.wait_until_settled(timeout=4.0)
        now = bot.driver.get_position(sid)
        results.append(abs(now - goal) <= OK_TICKS)
        print(f"    목표 {goal:>5} → 도착 {now:>5}  {'OK' if results[-1] else '어긋남'}")
    return all(results)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="로봇팔 연결 점검")
    ap.add_argument("--port", required=True, help='예: COM6, /dev/ttyACM0')
    ap.add_argument("--move", action="store_true", help="관절마다 약 5° 움직였다 돌아오기")
    ap.add_argument("--yes", action="store_true", help="관절마다 묻지 않기")
    args = ap.parse_args(argv)

    bot = Robot(args.port, speed=SPEED)
    try:
        if not show_status(bot) or not args.move:
            return 0
        bot.torque(True)                     # 지금 자세 그대로 잠금
        print("\n토크를 켰습니다 (지금 자세 유지). 관절을 하나씩 조금 움직입니다.")
        summary = {}
        for sid in JOINT_IDS:
            name = JOINT_NAMES[sid - 1]
            if not args.yes:
                ans = input(f"\n[{sid} {name}] Enter=움직이기, s=건너뛰기, q=그만: ").strip().lower()
                if ans == "q":
                    break
                if ans == "s":
                    continue
            else:
                print(f"\n[{sid} {name}]")
            summary[name] = wiggle(bot, sid)
        print("\n결과: " + ", ".join(f"{n} {'OK' if v else '확인 필요'}" for n, v in summary.items()))
    except KeyboardInterrupt:
        print("\n중단")
    finally:
        try:
            bot.stop()
        finally:
            bot.torque(False)
            print("토크 끔")
            bot.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
