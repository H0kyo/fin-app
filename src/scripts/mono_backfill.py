import sys

args = sys.argv[1:]
if "--src-path" in args:
    i = args.index("--src-path")
    sys.path.insert(0, args[i + 1])
    del args[i : i + 2]

from finapp.mono_backfill import main

main(args)
