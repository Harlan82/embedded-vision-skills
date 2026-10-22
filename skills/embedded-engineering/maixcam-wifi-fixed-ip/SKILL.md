---
name: maixcam-wifi-fixed-ip
description: 把 Sipeed MaixCAM / MaixCAM Pro / MaixCAM2 的无线地址固定成指定 IP（如 192.168.137.69），并让设备每次连接热点都主动请求该地址。适用于 PC 热点组网的双摄/多机项目，尤其是对端把相机地址硬编码的场景（如顶视 config.TOP_HOST）。当出现「相机固定 IP」「MaixCAM 固定 IP」「SSH 进设备改网络设置」「每次连热点都请求同一个 IP」「换了热点设备 IP 变了、顶视连不上主摄」「TOP_HOST 失配」等说法时使用。
agent_created: true
---

# MaixCAM 无线 IP 固定

把设备无线地址钉死成指定 IP，跨重启、跨重连生效。

## 开始前先明确两件事

1. **固定「拿到哪个地址」** —— 目标地址先确认，通常写死在某个 config 里（如 `TOP_HOST`）。
2. **还要固定「连哪个热点」** —— 设备常残留旧档案（例如手机热点）。
   只改地址不改档案，设备压根不上目标网段。**两件事都做完才算完成。**

## 步骤

### 1. 找到设备并连上

```bash
python scripts/find_device.py                            # 扫默认网段，看谁开了 22 端口
python scripts/find_device.py 192.168.137. --ports 22,8000
python scripts/devssh.py whoami                          # 连通性自检
DEV_HOST=192.168.137.69 python scripts/devssh.py run "uname -a"
```

判读：只开 22、ping 回 TTL=64 → 典型的精简 Linux（MaixCAM）。
有 80/443/8080 → 更像路由器或带 Web 端的摄像头，别改错对象。

| 机型 | 用户/密码 | 系统 |
|---|---|---|
| 一代 MaixCAM / MaixCAM Pro | `root` / `root` | buildroot（`/etc/init.d/S*`） |
| MaixCAM2 | `root` / `sipeed` | systemd |

优先走**有线/USB 通道**（设备 `usb0`/`usb1`）或主机名 `maixcam-xxxx.local`：
无线正是本轮要改的对象，不要让它成为唯一入口。

> Git Bash 下把远端路径当参数时必须前置 `MSYS_NO_PATHCONV=1`，否则 `/etc/...`
> 会被改写成 Windows 路径。脚本内置的另外两个坑见 `scripts/devssh.py` 顶部注释。

### 2. 认清结构（一代 buildroot）

无线全部由 `/etc/init.d/S30wifi` 驱动，配置落在 vfat 可写的 `/boot`：

| 文件 | 作用 |
|---|---|
| `/boot/wifi.sta` | 存在 = station 模式 |
| `/boot/wifi.ssid` + `/boot/wifi.pass` | 热点 SSID / 密码 |
| `/boot/wpa_supplicant.conf` | 存在则整份覆盖 `/etc/wpa_supplicant.conf`（只有要多网络时才用） |
| `/boot/wifi.nodhcp` | 存在 = 完全不跑 udhcpc（纯静态才用，会丢网关/DNS） |

原厂脚本逐行逻辑、`udhcpc` 行为、验证与备份命令见
`references/maixcam-buildroot-network.md`。

### 3. 改两处

**① 固定地址** —— 改 `/etc/init.d/S30wifi`：加 `WIFI_FIXED_IP` 常量、给 udhcpc 加 `-r`、
再加持续收敛的兜底补绑（代码直接照抄 `references/maixcam-buildroot-network.md` 第四节）。

**② 连对热点** —— 写 `/boot/wifi.ssid` 与 `/boot/wifi.pass`（内容不带换行）。

改完必做：推送 → `chmod 755` → `sh -n` 语法自检 → `/etc/init.d/S30wifi restart`
（restart 与开机同一条代码路径，可用于验证）。

### 4. 验证（五项缺一不可）

1. `ip -o addr show wlan0` 只有目标地址；
2. `ip route | grep default` 存在 → 是 DHCP 正式租约，而不是兜底绑出来的；
3. PC 侧 `ping <目标地址>` 通，且 `arp -a` 里 MAC 等于设备 `wlan0` 的 MAC；
4. PC 走该地址登录成功一次；
5. **重启设备**后再 ping 一次 → 通即持久化成立（顺手记 `uptime` 佐证）。

## 两个必踩的坑

1. **`udhcpc` 拿到租约时会抹掉同接口其它地址**
   （`/usr/share/udhcpc/default.script` 的 `renew|bound` 分支执行
   `/sbin/ifconfig $interface $ip $BROADCAST $NETMASK`）。
   → 补绑必须放在该事件之后、且持续收敛；一次性加完必然失效。

2. **Windows 热点会忽略「请求特定 IP」——但只在那个地址被旧 MAC 占着时。**
   判据：PC 侧 `arp -a` 里该地址类型为「静态」且 MAC 不是本设备。
   解法：重启一次移动热点清空 ICS 租约表，之后请求即被接受并记成静态租约。
   读热点密码、开关热点、PC 侧诊断顺序见 `references/windows-hotspot-ics.md`。

## 回滚与留档（改前必做）

- 备份 `/etc/init.d/S30wifi`、`/etc/wpa_supplicant.conf`、`/boot/wifi.*` 到
  `/root/net_backup_<日期>/`，并拉一份到 PC（busybox 的 `tar` 不支持 `-z`）。
- 在设备上放回滚脚本，两个档位：`iponly`（只撤固定 IP）/ `all`（连 WiFi 档案一起还原）。
- 备份路径与回滚用法写进交付记录，不要只留在对话里。

## 配套资源

| 路径 | 用途 |
|---|---|
| `scripts/devssh.py` | 非交互 SSH（run / sudo / pull / push），含 Windows 踩坑说明 |
| `scripts/find_device.py` | 扫网段定位设备、读 SSH 横幅判机型 |
| `references/maixcam-buildroot-network.md` | S30wifi 原厂逻辑、udhcpc 行为、可照抄的补丁、验证/备份命令 |
| `references/windows-hotspot-ics.md` | ICS 按请求分配的前提、WinRT 读密码与开关热点、PC 侧诊断顺序 |

## 收尾

对端硬编码地址的项目里，把「主相机 IP 是 DHCP 还是静态」这类悬而未决项一起结掉
（例如《对接协议总表》的「缺的资料」清单），并注明回滚方式——
否则下次换人换设备，同样的问题要从零再查一遍。
