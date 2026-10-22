# MaixCAM（一代，buildroot）网络机制参考

实测设备：`maixcam-17a4`，Linux 5.10.4 riscv64，busybox 1.36.1，root 密码 `root`。

## 一、启动顺序里与网络相关的脚本

`/etc/init.d/` 下相关项（SysV 风格，非 systemd）：

| 脚本 | 作用 |
|---|---|
| `S25wifimod` | 加载 WiFi 驱动并**清掉再插入内核模块**（重启 WiFi 时看到的 modinfo 告警出自这里） |
| `S30eth` / `S30gadget_nic` | 有线 / USB gadget 网卡（`usb0` / `usb1`） |
| `S30wifi` | **无线唯一入口**：关联 + 取地址（本技能改的就是它） |
| `S40network` | `ifup -a`，只拉起 `lo`（`/etc/network/interfaces` 里只有 lo） |
| `S49ntp` / `S50sshd` / `S70vsftpd` / `S80dnsmasq` | 时间、SSH、FTP、USB 侧 DHCP |

接口实测：

```
usb0  10.23.165.1/24     ← PC 侧对应网卡 10.23.165.100
usb1  10.23.164.1/24     ← PC 侧对应网卡 10.23.164.100
eth0  NO-CARRIER
wlan0 关联热点后才 up
```

`usb0/usb1` 是设备自己当 DHCP 服务器（dnsmasq）分配给 PC 的，
**是独立于无线的救援通道**：改无线配置改砸了也不会失去访问。

## 二、`/etc/init.d/S30wifi` 逻辑（原厂）

```sh
start() {
  if [ -e /boot/wifi.sta ]          # ← station 模式
  then
    if [ -e /boot/wpa_supplicant.conf ]
    then cp /boot/wpa_supplicant.conf /etc/wpa_supplicant.conf   # 整份覆盖，优先级最高
    else
      ssid=`cat /boot/wifi.ssid`; pass=`cat /boot/wifi.pass`
      wpa_passphrase "$ssid" "$pass" >> /etc/wpa_supplicant.conf  # 生成单网络配置
    fi
    wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant.conf
    if [ ! -e /boot/wifi.nodhcp ]   # ← 有这个文件就完全不跑 udhcpc
    then (udhcpc -i wlan0 -t 10 -T 1 -A 5 -b -p /run/udhcpc.wlan0.pid) &
    fi
  elif [ -e /boot/wifi.ap ]         # ← AP 模式（hostapd + udhcpd，10.x 网段）
  elif [ -e /boot/wifi.mon ]        # ← 监听模式
  fi
}
```

关键结论：

- `/boot` 是 vfat 可写分区（`/dev/mmcblk0p1`），`wifi.ssid` / `wifi.pass` / `wifi.sta` /
  `wifi.nodhcp` 都是**官方配置入口**，写进去重启即生效。
- `/boot/wifi.nodhcp` 存在时不取地址，网关/DNS 全丢 —— 纯静态方案才用，一般不要动。
- `stop()` 会 kill `wpa_supplicant` / `udhcpc` / `hostapd`，但**不清除 wlan0 上的地址**。
- `restart` = `stop` + `start`，与开机走同一条路径，可用于验证。

## 三、`udhcpc` 的行为（踩坑点）

busybox udhcpc 的 `-r IP` = "Request this IP address"，即 DISCOVER 里带 option 50。

事件脚本 `/usr/share/udhcpc/default.script` 的 `renew|bound` 分支第一步就是：

```sh
/sbin/ifconfig $interface $ip $BROADCAST $NETMASK
```

`ifconfig <iface> <ip>` 会**重设地址、抹掉同一接口上其它地址**。
后果：如果在 udhcpc 之前/期间手动 `ip addr add`，地址会被冲掉。

→ 补绑固定地址必须放在这个事件之后，并且**持续收敛**（循环若干秒重试），不能一次性完成。

`-r` 请求能否被满足取决于服务器（见 `windows-hotspot-ics.md`），所以「请求 + 兜底」两手都要。

## 四、本技能用到的改动（可直接照抄）

```sh
# 文件头
WIFI_FIXED_IP="192.168.137.69"

# start() 中的地址获取段
(udhcpc -i wlan0 -r "${WIFI_FIXED_IP}" -t 10 -T 1 -A 5 -b -p /run/udhcpc.wlan0.pid) &
(
    i=0
    while [ $i -lt 90 ]
    do
        ip addr show wlan0 | grep -q "inet ${WIFI_FIXED_IP}/" || \
            ip addr add ${WIFI_FIXED_IP}/24 dev wlan0 2>/dev/null
        sleep 2
        i=$((i + 1))
    done
) &
```

外加 `/boot/wifi.ssid` + `/boot/wifi.pass` 指向目标热点（无换行写入）。

## 五、验证命令

```bash
sh -n /etc/init.d/S30wifi                  # 语法自检（改完必做）
/etc/init.d/S30wifi restart
sleep 25
ip -o addr show wlan0                      # 应只有目标地址
ip route | grep default                    # 有 default 说明是 DHCP 正式租约
wpa_cli -i wlan0 status | head             # wpa_state=COMPLETED / ssid / ip_address
cat /sys/class/net/wlan0/address           # 与 PC 侧 arp -a 的 MAC 对照
grep -n WIFI_FIXED_IP /etc/init.d/S30wifi  # 确认改动在设备上
```

持久化验证：`(sleep 2; reboot) &` → 等 15~20 s → ping 目标地址。
（设备重启时 `usb0/usb1` 也会短暂断开，PC 侧那条链路会掉一会儿，属正常。）

## 六、留档与回滚（改前必做）

```bash
B=/root/net_backup_$(date +%Y%m%d); mkdir -p $B
cp -a /etc/init.d/S30wifi $B/S30wifi.bak
cp -a /etc/wpa_supplicant.conf $B/wpa_supplicant.conf.bak
cp -a /etc/network/interfaces $B/interfaces.bak
cp -a /boot/wifi.ssid /boot/wifi.pass /boot/wifi.sta $B/
ls -la /boot > $B/boot.listing.txt
# busybox 的 tar 不支持 -z，用 tar cf 或不打包
```

回滚脚本给两个档位（`iponly` = 只撤固定 IP；`all` = 连 WiFi 档案一起还原），
并同时把备份拉一份到 PC。
