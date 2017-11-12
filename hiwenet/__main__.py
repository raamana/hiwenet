from sys import version_info

if version_info.major==2 and version_info.minor==7:
    from pairwise_dist import run_cli
elif version_info.major > 2:
    from hiwenet.pairwise_dist import run_cli
else:
    raise NotImplementedError('hiwenet supports only 2.7.13 or 3+. Upgrade to Python 3+ is recommended.')

def main():
    "Entry point."

    run_cli()

if __name__ == '__main__':
    main()
