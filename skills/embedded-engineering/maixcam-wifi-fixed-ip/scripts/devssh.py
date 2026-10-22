# -*- coding: utf-8 -*-
"""devssh.py —— 对 MaixCAM 一类设备做非交互 SSH 操作（基于 paramiko）。

为什么不直接用系统的 ssh 命令：
    密码交互在脚本/沙箱里不可靠；SSH_ASKPASS 在 Windows 上要求助手是可直接执行的
    程序（.sh 无法被 CreateProcess 执行）。paramiko 可以完全非交互、可脚本化。

用法：
    python devssh.py run  "<cmd>"            执行命令
    python devssh.py sudo "<cmd>"            用密码 sudo 执行
    python devssh.py pull <remote> <local>   读远端文件到本地
    python devssh.py push <local> <remote>   写本地文件到远端
    python devssh.py whoami                  连通性自检（内核/身份/主机名）

连接参数（环境变量，均有默认值）：
    DEV_HOST  默认 192.168.137.69
    DEV_PORT  默认 22
    DEV_USER  默认 root
    DEV_PW    不给则按候选表依次尝试

安装依赖：
    <python> -m venv <venv> && <venv>/Scripts/python -m pip install paramiko

Windows + Git Bash 两个实测坑：
    1) 远端路径当命令行参数时必须前置 MSYS_NO_PATHCONV=1，
       否则 /etc/init.d/S30wifi 会被改写成 C:/Users/.../etc/init.d/S30wifi。
    2) 部分设备的 SFTP 子系统不可用（sftp.get 报 ENOENT，exec 正常），
       所以 pull/push 走 exec + stdin，而不是 sftp。
"""
import io
import os
import sys

import paramiko

HOST = os.environ.get("DEV_HOST", "192.168.137.69")
PORT = int(os.environ.get("DEV_PORT", "22"))
USER = os.environ.get("DEV_USER", "root")
# 出厂/常见默认密码，非用户真实凭据。真实密码请用 DEV_PW 环境变量覆盖。
# 注意：若设备长期使用候选表中的密码且网络可达，等同于无鉴权，务必改掉。
CANDIDATES = ["root", "sipeed", "123456"]

_CUR_PW = ""


def _passwords():
    pw = os.environ.get("DEV_PW")
    return [pw] if pw else CANDIDATES


def connect():
    """按候选密码表连接，返回 (client, 生效密码)。"""
    last = None
    for pw in _passwords():
        cli = paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            cli.connect(
                HOST, port=PORT, username=USER, password=pw,
                timeout=8, banner_timeout=8, auth_timeout=8,
                look_for_keys=False, allow_agent=False,
            )
        except paramiko.AuthenticationException as e:
            last = e
            try:
                cli.close()
            except Exception:
                pass
            continue
        return cli, pw
    raise SystemExit("[X] 认证失败（试过 %s）：%s" % (CANDIDATES, last))


def run(cli, cmd, sudo=False, stdin_data=None, timeout=90):
    if sudo:
        cmd = "echo %s | sudo -S -p '' sh -c '%s'" % (
            "'" + _CUR_PW + "'", cmd.replace("'", "'\"'\"'"))
    si, so, se = cli.exec_command(cmd, timeout=timeout)
    if stdin_data is not None:
        si.write(stdin_data)
        si.flush()
        try:
            si.channel.shutdown_write()
        except Exception:
            pass
    out = so.read().decode("utf-8", "replace")
    err = se.read().decode("utf-8", "replace")
    return out, err, so.channel.recv_exit_status()


def _q(path):
    return "'%s'" % path.replace("'", "'\"'\"'")


def main():
    global _CUR_PW
    action = sys.argv[1] if len(sys.argv) > 1 else "whoami"
    arg = sys.argv[2] if len(sys.argv) > 2 else ""

    cli, pw = connect()
    _CUR_PW = pw
    print("[i] SSH %s@%s:%d (password=%s)" % (USER, HOST, PORT, pw))

    if action == "whoami":
        out, err, code = run(cli, "uname -a; id; hostname")
    elif action == "run":
        out, err, code = run(cli, arg)
    elif action == "sudo":
        out, err, code = run(cli, arg, sudo=True)
    elif action == "pull":
        remote, local = sys.argv[2], sys.argv[3]
        out, err, code = run(cli, "cat " + _q(remote))
        if code == 0:
            with io.open(local, "wb") as f:
                f.write(out.encode("utf-8"))
            out = "[i] pulled %s -> %s (%d bytes)" % (
                remote, local, len(out.encode("utf-8")))
    elif action == "push":
        local, remote = sys.argv[2], sys.argv[3]
        with io.open(local, "rb") as f:
            data = f.read()
        out, err, code = run(cli, "cat > " + _q(remote), stdin_data=data)
        if code == 0:
            chk, _, _ = run(cli, "wc -c < " + _q(remote))
            out = "[i] pushed %s -> %s (local %d, remote %s bytes)" % (
                local, remote, len(data), chk.strip())
    else:
        raise SystemExit("unknown action: %s" % action)

    cli.close()
    if out and not out.endswith("\n"):
        out += "\n"
    sys.stdout.write("===== stdout =====\n" + out)
    if err.strip():
        sys.stdout.write("===== stderr =====\n" + err + "\n")
    sys.stdout.write("[exit] %d\n" % code)
    sys.exit(0 if code == 0 else 1)


if __name__ == "__main__":
    main()
