# Windows 移动热点（ICS）行为参考 + 从 PC 侧操作

场景：PC 开「移动热点」当 AP（`192.168.137.1`），MaixCAM 等设备连上来取地址。
设备侧固定 IP 之后，**能否真的拿到那个地址，取决于 ICS 的 DHCP 行为**。

## 一、ICS 的 DHCP 会「按请求分配」，但有前提

实测：

| 情况 | 结果 |
|---|---|
| 目标地址被 ICS 绑给了**别的 MAC**（旧设备） | 请求被忽略，发别的地址（实测发了 `.55`），日志形如 `select for 192.168.137.55` |
| 目标地址**空闲** | 请求被接受，日志形如 `select for 192.168.137.69`，且 ICS 记成静态租约 |
| 租期 | `lease time 604800`（7 天） |

PC 侧 `arp -a` 里设备地址的「类型」字段可判读状态：

```
接口: 192.168.137.1 --- 0xd
  192.168.137.69        aa-bb-cc-dd-ee-01     静态   ← 旧设备的租约残留，会挡住你的请求
  192.168.137.55        aa-bb-cc-dd-ee-02     动态   ← 实际拿到的地址
```

「静态」= ICS 为它服务过的租约记的保留项。若 MAC 不是目标设备的 MAC，
就说明这个地址被旧租约占着 → **请求必被忽略，PC 也 ping 不到该地址**。

**解法**：重启一次移动热点，ICS 租约表清空，请求立刻被接受。
（`arp -d` / `netsh interface ip delete arpcache` 都需要管理员权限，通常不可用。）

## 二、读热点 SSID / 密码（PowerShell 5.1 + WinRT）

结果写文件再读 —— PowerShell 的 stdout 在本机不回传。

```powershell
# hotspot_pw.ps1（保持纯 ASCII，避免 PS 5.1 按 GBK 读取 UTF-8 报「字符串缺少终止符」）
$out = Join-Path $env:TEMP "hotspot_pw.txt"
try {
    [void][Windows.Networking.Connectivity.NetworkInformation, Windows.Networking.Connectivity, ContentType = WindowsRuntime]
    [void][Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager, Windows.Networking.NetworkOperators, ContentType = WindowsRuntime]
    $conn = [Windows.Networking.Connectivity.NetworkInformation]::GetInternetConnectionProfile()
    $mgr = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager]::CreateFromConnectionProfile($conn)
    $cfg = $mgr.GetCurrentAccessPointConfiguration()
    $lines = @("SSID = " + $cfg.Ssid, "PASS = " + $cfg.Passphrase, "band = " + $cfg.Band,
               "state = " + $mgr.TetheringOperationalState)
}
catch { $lines = @("ERR: " + $_.Exception.Message) }
$lines | Out-File -FilePath $out -Encoding UTF8
```

## 三、开关热点（用于清租约表）

```powershell
$mgr.StopTetheringAsync()  | Out-Null     # 注意：没有 StopTethering() 这个方法
Start-Sleep -Seconds 5
[void]$mgr.StartTetheringAsync()
Start-Sleep -Seconds 14
$mgr.TetheringOperationalState            # 期望回到 On
```

状态会经历 `On → InTransition → Off → On`。
若停在 `Off`，再 `StartTetheringAsync()` 一次即可。

也可以从别处判读热点是否在跑：`netsh wlan show interfaces` 看本机 WLAN 连接，
扫热点 SSID 用设备侧 `iw dev wlan0 scan | grep SSID`（AP 的 SSID 就是
`LAPTOP-XXXX NNNN` 这种系统默认名）。

## 四、诊断顺序（PC → 设备方向不通时）

1. `ping <设备地址>`：不通先怀疑 ARP，而不是设备。
2. `arp -a` 看该地址的 MAC 与「类型」：与设备 `cat /sys/class/net/wlan0/address` 对照。
3. 设备侧确认地址真的在：`ip -o addr show wlan0`（可能是 secondary 地址）。
4. 设备侧 `iptables -L INPUT -n`：默认是 `policy ACCEPT`、无规则，一般不是防火墙问题。
5. 若 MAC 不符 → 走第一节的「重启热点清租约」。
6. **注意**：只有 PC 受旧 ARP 影响；另一台设备（如顶视）做全新 ARP 时能正常解析到，
   所以「PC ping 不通」≠「顶视连不上」。

## 五、本机（Windows）操作注意

- `reg.exe` 被本机安全策略拉黑，读注册表这条路走不通 → 用 WinRT 或 `netsh`。
- Git Bash 会把 `/xxx` 形式参数改成 Windows 路径 → 前置 `MSYS_NO_PATHCONV=1`。
- `timeout` 在 Git Bash 里会命中 Windows 的 `timeout.exe`，报
  「默认选项不允许超过 1 次」→ 用 `curl -m <秒>` 或 Python socket 自带超时。
- PowerShell 工具不回传 stdout → 一律「写文件 + Read」。
- `ipconfig //all` 也会被路径转换吃掉 → 读全部网卡用
  `netsh interface ip show config`（输出是 UTF-8，不要再 iconv 成 GBK）；
  需要脚本判读时用 Python socket 直接连更稳。
- 想探某地址是不是 SSH 主机：用 Python socket 读横幅（`SSH-2.0-OpenSSH_9.6`、
  `SSH-2.0-dropbear` 等），比 curl 的 telnet 模式在 Windows 上更可控。
