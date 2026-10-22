# -*- coding: utf-8 -*-
"""find_device.py —— 扫网段找 MaixCAM：哪些地址开了 22/80/8000，SSH 横幅是什么。

用途：
  - 设备换了地址、不知道现在在哪；
  - 确认某地址到底是相机还是路由器/AP（看 SSH 横幅：MaixCAM 一代是
    OpenSSH 或 dropbear，PC/路由器另有特征）；
  - 确认「疑似残留 ARP 记录」的那个地址上是否真的有设备。

用法：
    python find_device.py                      # 扫默认网段
    python find_device.py 192.168.137. 10.23.165.
    python find_device.py 192.168.137. --ports 22,8000

默认网段 = 常见相机网段（PC 热点 192.168.137.0/24、USB gadget 10.23.165/164.0/24）。

判读要点：
  - 只回 TTL=64 的 ping + 只有 22 端口 → 典型的精简 Linux（MaixCAM）。
  - 有 80/443/8080 → 更可能是路由器/摄像头 Web 端。
  - 扫描期间无响应的地址不代表不存在，可能只是关了 ICMP/端口。
"""
import concurrent.futures
import socket
import sys

DEFAULT_PREFIXES = ("192.168.137.", "10.23.165.", "10.23.164.")


def probe(ip, ports, tmo=0.4):
    hits = []
    for p in ports:
        s = socket.socket()
        s.settimeout(tmo)
        try:
            s.connect((ip, p))
            banner = b""
            try:
                s.settimeout(1.5)
                banner = s.recv(160)
            except Exception:
                pass
            hits.append((p, banner.decode("utf-8", "replace").strip()))
        except Exception:
            pass
        finally:
            try:
                s.close()
            except Exception:
                pass
    return ip, hits


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    prefixes = tuple(args) if args else DEFAULT_PREFIXES
    ports = (22, 23, 80, 8000)
    if "--ports" in sys.argv:
        ports = tuple(int(x) for x in sys.argv[sys.argv.index("--ports") + 1].split(","))

    for prefix in prefixes:
        ips = [prefix + str(i) for i in range(1, 255)]
        print("===== 扫描 %s0/24  端口 %s =====" % (prefix, list(ports)))
        found = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=80) as ex:
            for ip, hits in ex.map(lambda i: probe(i, ports), ips):
                for p, b in hits:
                    found += 1
                    print("  %-16s :%-5d %s" % (ip, p, b if b else "(已连接，无横幅)"))
        if not found:
            print("  (无主机响应)")
        print()


if __name__ == "__main__":
    main()
