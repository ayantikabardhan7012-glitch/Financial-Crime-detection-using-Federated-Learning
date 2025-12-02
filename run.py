import subprocess

scripts = [
    "python server.py",
    "python new_client1.py",
    "python new_client2.py",
    "python new_client3.py"
]

for cmd in scripts:
    subprocess.Popen(["start", "cmd", "/k", cmd], shell=True)
