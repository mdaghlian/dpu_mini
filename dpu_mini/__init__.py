# Check for requirements:
import os
import subprocess
# ************************** SPECIFY FS_LICENSE HERE **************************
# [1] Freesurfer, freeview
fs_cmd_list = ['freeview']
for cmd in fs_cmd_list:
    cmd_out = subprocess.getstatusoutput(f"command -v {cmd}")[1]
    if cmd_out=='':
        print(f'Could not find path for {cmd}, is freesurfer accessible from here?')

# os.environ['FS_LICENSE'] = '/data1/projects/dumoulinlab/Lab_members/Marcus/programs/linescanning/misc/license.txt'
if 'FS_LICENSE' in os.environ.keys():
    if not os.path.exists(os.environ['FS_LICENSE']):
        print('Could not find FS_LICENSE, set using os.environ above')
else:
    print('Could not find FS_LICENSE')
    print('Uncomment line below and specify path to FS_LICENSE')


# ************************** CHECK FOR SUBJECTS DIR **************************
if "SUBJECTS_DIR" not in os.environ.keys(): # USED FOR DEFAULT FS DIR
    print('SUBJECTS_DIR not found in os.environ')
    print('Adding empty string...')
    os.environ['SUBJECTS_DIR'] = ''
    