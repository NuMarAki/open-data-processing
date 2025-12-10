import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

if __name__ == '__main__':
    required_packages = [
        'pandas',
        'pyarrow',
        'numpy',
        'matplotlib',
        'glob2'
    ]
    
    for package in required_packages:
        install(package)