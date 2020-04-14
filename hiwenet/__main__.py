from sys import version_info

if version_info.major > 2:
    from hiwenet.pairwise_dist import run_cli
else:
    raise NotImplementedError('hiwenet supports only Python 3 or higher!')

def main():
    "Entry point."

    run_cli()

if __name__ == '__main__':
    main()
