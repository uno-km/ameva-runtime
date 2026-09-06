"""
Master Cluster Watchdog Daemon (호스트 PC 마스터 컨트롤 타워)
- 관리 대상: Galaxy S21, A53, A35, S20, S7
- 기능:
  1. 원격 ADB 연결 상시 유지 및 자동 재연결
  2. Termux 프로세스 실시간 생존 감시 (Watchdog)
  3. 다운 감지 시 10초 이내 원격 강제 부활 (am start + wake lock + sshd 복구)
  4. 기기별 배터리/온도/CPU 상태 텔레메트리 대시보드
"""

import subprocess
import time
import sys
import os

DEVICES = [
    {"id": "100.83.82.60:5555", "name": "Galaxy S21 5G", "type": "tailscale"},
    {"id": "100.77.47.37:5555", "name": "Galaxy A53 5G", "type": "tailscale"},
    {"id": "100.106.251.21:5555", "name": "Galaxy A35", "type": "tailscale"},
    {"id": "100.106.99.81:5555", "name": "Galaxy S20 5G", "type": "tailscale"},
    {"id": "ce11160b5ab0b61305", "name": "Galaxy S7", "type": "usb"}
]

def run_adb(args, timeout=8):
    try:
        res = subprocess.run(
            ["adb"] + args,
            capture_output=True,
            timeout=timeout
        )
        stdout = res.stdout.decode('utf-8', errors='ignore').strip()
        stderr = res.stderr.decode('utf-8', errors='ignore').strip()
        return stdout, stderr, res.returncode
    except Exception as e:
        return "", str(e), -1

def ensure_connections():
    out, _, _ = run_adb(["devices"])
    for dev in DEVICES:
        if dev["type"] == "tailscale":
            if dev["id"] not in out or "offline" in out:
                run_adb(["connect", dev["id"]], timeout=4)

def check_and_revive_device(dev):
    dev_id = dev["id"]
    name = dev["name"]
    
    out, _, _ = run_adb(["devices"])
    if dev_id not in out:
        return f"[DISCONNECTED] {name} 오프라인"
    if f"{dev_id}\tunauthorized" in out:
        return f"[UNAUTHORIZED] {name} 화면 승인 필요"
    if f"{dev_id}\toffline" in out:
        run_adb(["disconnect", dev_id])
        run_adb(["connect", dev_id])
        return f"[RECONNECTING] {name} 재연결 시도 중"

    stdout, stderr, code = run_adb(["-s", dev_id, "shell", "pidof com.termux || pgrep -f com.termux"])
    pids = stdout.strip()

    status_msg = ""
    if not pids:
        print(f"\n[ALERT] {name} ({dev_id}) Termux 다운 감지! 즉시 원격 부활 명령 송출...")
        run_adb(["-s", dev_id, "shell", "am", "start", "-n", "com.termux/.app.TermuxActivity"])
        time.sleep(2)
        run_adb(["-s", dev_id, "shell", "am", "startservice", "--user", "0",
                 "-n", "com.termux/com.termux.app.RunCommandService",
                 "-a", "com.termux.RUN_COMMAND",
                 "--es", "com.termux.RUN_COMMAND_PATH", "/data/data/com.termux/files/usr/bin/bash",
                 "--esa", "com.termux.RUN_COMMAND_ARGUMENTS", "-c,termux-wake-lock; sshd"])
        status_msg = f"[REVIVED] {name} Termux 강제 기동 완료"
    else:
        bat_out, _, _ = run_adb(["-s", dev_id, "shell", "dumpsys battery"])
        level = "N/A"
        temp = "N/A"
        for line in bat_out.splitlines():
            line_str = line.strip()
            if line_str.startswith("level:"):
                level = line_str.split("level:")[1].strip() + "%"
            elif line_str.startswith("temperature:"):
                try:
                    raw_temp = float(line_str.split("temperature:")[1].strip()) / 10.0
                    temp = f"{raw_temp:.1f}C"
                except:
                    pass
        status_msg = f"[ALIVE] {name:<14} (PID: {pids.split()[0]:<6}) | 배터리: {level:<5} | 온도: {temp:<6} | ADB 5555 직결"

    return status_msg

def main():
    print("=" * 80)
    print("  [AMEVA] Master Cluster Watchdog Daemon (호스트 PC 마스터 컨트롤 타워)")
    print("=" * 80)
    print(f"[*] 총 {len(DEVICES)}개 기기 텔레메트리 헬스체크 수행\n")
    
    ensure_connections()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- [Telemetry Health Check: {timestamp}] ---")
    for dev in DEVICES:
        result = check_and_revive_device(dev)
        print(f"  * {result}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
