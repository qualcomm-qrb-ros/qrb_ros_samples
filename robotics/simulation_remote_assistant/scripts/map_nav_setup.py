# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
# SPDX-License-Identifier: BSD-3-Clause-Clear

import subprocess
import time
import signal
import sys

procs = []   

def launch_ros_launch(pkg, file):
    proc = subprocess.Popen(['ros2', 'launch', pkg, file, 'use_sim_time:=true'])
    procs.append(proc)
    return proc

def launch_ros_run(pkg, executable):
    proc = subprocess.Popen(['ros2', 'run', pkg, executable])
    procs.append(proc)
    return proc

def kill_all():
    print('\nKilling all launched processes...')
    for p in procs:
        if p and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                p.kill()
    print('All child processes killed.')

def sigint_handler(sig, frame):
    kill_all()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)
    try:
        print('Launching cartographer_ros for mapping...')
        carto_proc = launch_ros_launch('cartographer_ros', 'qrb_2d_lidar_slam.launch.py')
        time.sleep(5)  

        print('Running build_map_node...')
        buildmap_proc = launch_ros_run('simulation_remote_assistant', 'build_map_node')
        buildmap_proc.wait()
        print('Mapping finished.')

        print('Restarting cartographer_ros...')
        if carto_proc and carto_proc.poll() is None:
            carto_proc.terminate()
            carto_proc.wait(timeout=5)
        time.sleep(2)
        carto_proc2 = launch_ros_launch('cartographer_ros', 'qrb_2d_lidar_slam.launch.py')
        time.sleep(5)

        print('Running nav_preparation_node...')
        navprep_proc = launch_ros_run('simulation_remote_assistant', 'nav_preparation_node')
        navprep_proc.wait()

    except KeyboardInterrupt:
        kill_all()
        print('User interrupted, shutting down...')
    except Exception as e:
        print(f'Exception occurs: {e}')
        kill_all()
    finally:
        kill_all()