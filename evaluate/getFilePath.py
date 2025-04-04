import os
import paramiko
import socket
# 本地文件路径

# 服务器 IP 列表
servers = [
    "222.200.185.120",
    "222.200.185.183",
    "222.200.185.186",
    "222.200.185.9",
    "222.200.185.138",
    "222.200.185.136",
    "222.200.185.67"
]

# 统一的 SSH 账号和密码
username = "wentao"
password = "wentao@123"

# 获取本机 IP 地址
def get_local_ip():
    """获取本机 IP 地址"""
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except Exception as e:
        print(f"获取本机 IP 失败: {e}")
        return None

def file_exists(filepath):
    """检查本地文件是否存在"""
    return os.path.exists(filepath)


def download_file(server_ip, remote_path, local_path):
    """尝试从服务器下载文件"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server_ip, username=username, password=password)

        sftp = ssh.open_sftp()
        # 先检查远程文件是否存在
        try:
            sftp.stat(remote_path)  # 如果文件不存在，会抛出异常
        except FileNotFoundError:
            print(f"远程文件不存在: {server_ip}:{remote_path}")
            sftp.close()
            ssh.close()
            return False

        # 创建本地文件夹（如果不存在）
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        # 开始下载
        print(f"从 {server_ip} 下载文件 {remote_path} -> {local_path}")
        sftp.get(remote_path, local_path)
        sftp.close()
        ssh.close()

        return True
    except Exception as e:
        print(f"从 {server_ip} 下载失败: {e}")
        return False


def get_local_or_remote_file_path(local_file_path):
    """获取本地文件，如果不存在则尝试从服务器下载"""
    if file_exists(local_file_path):
        print(f"文件已存在: {local_file_path}")
        return local_file_path
    local_ip = get_local_ip()
    filtered_servers = [ip for ip in servers if ip != local_ip]
    print("本地文件不存在，尝试从服务器下载...")
    for server in filtered_servers:
        print(f"尝试从 {server} 下载...")
        if download_file(server, local_file_path, local_file_path):
            print(f"成功从 {server} 下载文件: {local_file_path}")
            return local_file_path

    print("所有服务器均无法下载文件")
    return None


if __name__ == "__main__":
    local_file_path = "/home/wentao/xzw/checkpoint/083_20250222195607/checkpoints/model_epoch_all_260000.pth"
    file_path = get_local_or_remote_file_path(local_file_path)
    if file_path:
        print(f"文件路径: {file_path}")
    else:
        print("无法获取文件")
