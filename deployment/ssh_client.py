"""Workspace SSH transport. Password arrives only through the child environment."""
import argparse
import base64
import hashlib
import os
from pathlib import Path
import sys
import time

import paramiko


class RobotHostPolicy(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        known=[existing.asbytes() for keys in client.get_host_keys().values() for existing in keys.values()]
        if known and key.asbytes() not in known:
            raise paramiko.SSHException('New robot address does not match the recorded robot host key')
        client.get_host_keys().add(hostname,key.get_name(),key)
        client.save_host_keys(client._host_keys_filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--known-hosts', required=True)
    parser.add_argument('--timeout', type=int, default=90)
    parser.add_argument('--command')
    parser.add_argument('--script')
    parser.add_argument('--upload')
    parser.add_argument('--download')
    parser.add_argument('--destination')
    args = parser.parse_args()
    password = os.environ.pop('MENTORPI_SSH_PASSWORD')
    known_hosts = Path(args.known_hosts)
    known_hosts.touch(exist_ok=True)
    client = paramiko.SSHClient()
    client.load_host_keys(str(known_hosts))
    # First use records the key; subsequent connections reject changed keys.
    client.set_missing_host_key_policy(RobotHostPolicy())
    try:
        client.connect(args.host, username=args.user, password=password, timeout=12,
                       auth_timeout=15, banner_timeout=15, allow_agent=False, look_for_keys=False)
        key = client.get_transport().get_remote_server_key()
        fingerprint = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
        print(f'SSH host key SHA256:{fingerprint}', file=sys.stderr, flush=True)
        if args.upload:
            with client.open_sftp() as sftp:
                sftp.put(args.upload, args.destination)
            print(f'Uploaded {Path(args.upload).name} to {args.destination}', flush=True)
            return 0
        if args.download:
            Path(args.destination).parent.mkdir(parents=True,exist_ok=True)
            with client.open_sftp() as sftp:
                sftp.get(args.download,args.destination)
            print(f'Downloaded {args.download}',flush=True)
            return 0
        command = args.command or ('python3 -' if args.script else 'true')
        stdin, stdout, stderr = client.exec_command(command, timeout=args.timeout)
        if args.script:
            stdin.write(Path(args.script).read_text(encoding='utf-8'))
        stdin.channel.shutdown_write()
        channel = stdout.channel
        deadline = time.monotonic() + args.timeout
        while True:
            if channel.recv_ready():
                sys.stdout.buffer.write(channel.recv(65536))
                sys.stdout.buffer.flush()
            if channel.recv_stderr_ready():
                sys.stderr.buffer.write(channel.recv_stderr(65536))
                sys.stderr.buffer.flush()
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                return channel.recv_exit_status()
            if time.monotonic() > deadline:
                channel.close()
                raise TimeoutError(f'Remote command exceeded {args.timeout} seconds')
            time.sleep(.05)
    finally:
        client.close()


if __name__ == '__main__':
    raise SystemExit(main())
