# arm — 로봇팔 제어

담당: **A (예은, @ye2un96-tech)**

SO-ARM101(Feetech STS3215) 조립과 동작을 맡습니다.

## 할 일
- 로봇팔 조립, 서보 모터 각도 조정, 캘리브레이션
- 대기 자세, 컵 집기·옮기기, 뚜껑·빨대 분리, 음료 기울여 붓기, 빈 컵 버리기 동작
- 그리퍼 힘 조절
- 자세·좌표·속도·그리퍼 힘은 설정 파일에 모으기 (CLAUDE.md 4.2)
- 실제 장치 / fake 장치를 바꿔 쓸 수 있게 만들기 (CLAUDE.md 4.4)

## 동작 순서

```text
home → 컵 잡기(open → grip_cup, 컵 중간~아래를 hold) → lift_cup
     → hook_ready → hook_engage → hook_pull → hook_clear
                                        (고정 고리에 뚜껑 가장자리를 걸고, 내리며 기울여 벗김 → 플라스틱함)
     → pour_ready → pour_tilt → pour_ready (배수통에 음료·얼음)
     → paper_bin, 그리퍼 열기               (종이류함에 빈 컵)
     → home
```

### 뚜껑 제거: 고리 방식

2026-11-27 실제 영화관 컵으로 시험한 결과, 컵 윗부분을 눌러 뚜껑을 헐겁게 하는 방식은 실패했습니다. 그래서 3D 프린터로 만든 **고리**를 고정해 두고 뚜껑을 벗기는 방식으로 바꿨습니다.

1. `hook_ready`: 고리 끝이 뚜껑 가장자리(스커트)보다 1~2cm 아래, 컵 벽 바로 옆에 오게 합니다.
2. `hook_engage`: 컵을 살짝 내려 스커트 아래 가장자리를 고리 끝에 얹습니다.
3. `hook_pull`: 컵을 더 내리면서 고리 반대쪽으로 기울입니다. 병따개처럼 뚜껑이 한쪽부터 벗겨집니다.
4. `hook_clear`: 컵을 고리에서 떼면 뚜껑과 빨대가 아래 플라스틱함으로 떨어집니다.

고리 설계안은 다음과 같습니다. 정확한 치수는 실제 컵을 재서 정합니다.
- 고리 끝은 컵 벽 쪽으로 얇아지는 쐐기 모양입니다.
- 컵 둘레를 따라 휜 원호 모양입니다.
- 받침대에 긴 구멍(슬롯)을 내서 높이를 조절합니다.

걸고 당기는 구간은 `speed.hook` 으로 느리게 움직입니다. 방법이 또 바뀌면 `cup_arm.py` 의 `remove_lid` 만 고칩니다.

## 파일

| 파일 | 내용 |
|---|---|
| `config.yaml` | 포트, 속도, 그리퍼 %, 기다리는 시간, 자세 이름 — **실측값은 모두 여기** |
| `cup_arm.py` | 단계별 동작 함수 (`pick_cup`, `remove_lid`, `pour`, `drop_cup`, `process_cup`, `safe_stop`) |
| `run_cup.py` | 실행. Ctrl+C 로 멈추면 대기 자세 → 토크 끄기 |
| `teach.py` | 자세 저장: 토크를 끄고 손으로 옮긴 뒤 Enter |
| `data/poses.json` | 저장된 자세 (teach.py 가 만듦, Git 에 올림) |
| `soarm/`, `model/` | 로봇팔 SDK — 아래 "가져온 코드" 참고 |
| `tests/` | 가상 팔로 돌리는 시험 |

## 사용법

필요한 패키지: `pyserial`, `numpy==1.26.4`, `pyyaml` (시험은 `pytest`)

```bash
cd arm
python run_cup.py --port fake              # 가상 팔로 흐름 확인 (자세가 없으면 건너뜀)
python teach.py --port /dev/ttyACM0        # 자세 9개를 차례로 저장
python teach.py --port /dev/ttyACM0 --list # 저장 상태 보기
python run_cup.py --port /dev/ttyACM0 --slow            # 처음엔 느리게
python run_cup.py --port /dev/ttyACM0 --step lid        # 한 단계만 (home, pick, lid, pour, drop)
python run_cup.py --port /dev/ttyACM0 --count 10        # 10회 연속 (성공률 기록용)
python -m pytest tests -q                  # 시험
```

> 실물로 처음 돌릴 때는 물로 시험하고, Pi·배선을 배수통에서 멀리 두세요 (CLAUDE.md 4.3).
> `safe_stop` 은 토크를 끄므로 팔이 힘이 빠져 내려옵니다. 대기 자세(home)를 팔이 내려와도 안전한 자세로 저장하세요.

## 가져온 코드

`soarm/`(SDK)과 `model/`(kinematics.json, so101.urdf)은 이전 실습 저장소 `th2102da/so101-lab` 의 커밋 `c3c3cec` 에서 수정 없이 복사했습니다 (CLAUDE.md 4.6). 고칠 일이 생기면 이 저장소에서 고친 내용을 커밋 메시지에 적습니다.

캘리브레이션 파일 `data/calibration.json` 은 이 팔에서 직접 만든 뒤 넣습니다. 없으면 SDK 기본 범위를 씁니다 (TODO: 실기 확인 필요).
