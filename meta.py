import re
import sys


def parse_version(git_version):
    m = re.search('v(.+)', git_version)
    return m.group(1)


def update_version(git_version):
    v = parse_version(git_version)
    with open('reinforcement/_version.py', 'w') as f:
        f.write(f"__version__ = '{v}'\n")


if __name__ == "__main__":
    update_version(sys.argv[1])
