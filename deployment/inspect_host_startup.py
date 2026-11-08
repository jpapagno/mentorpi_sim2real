"""Bounded host startup evidence, using structured subprocess arguments."""
import json
import subprocess

commands={
 'journal':['journalctl','-u','start_node.service','-n','80','--no-pager','-o','cat'],
 'driver_state':['docker','exec','MentorPi','sh','-c',"ps -eo pid,stat,comm,args | grep -E 'ldlidar|ros_robot_controller|ekf_node' | grep -v grep; ls -l /proc/6642/fd 2>/dev/null"],
 'log_locations':['docker','exec','MentorPi','sh','-c','ls -ld /home/ubuntu/.ros/log /root/.ros/log /tmp/*.log 2>/dev/null'],
}
result={}
for key,command in commands.items():
 try:
  run=subprocess.run(command,capture_output=True,text=True,timeout=20)
  result[key]={'exit':run.returncode,'out':run.stdout,'err':run.stderr}
 except Exception as exc:result[key]={'error':str(exc)}
print(json.dumps(result,indent=2),flush=True)
