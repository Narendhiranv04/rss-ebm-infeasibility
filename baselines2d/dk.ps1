# Run a command in the Linux container with "D:\RRC\New folder" mounted at /work.
#   .\dk.ps1 python scripts/parity.py 6
param([Parameter(ValueFromRemainingArguments = $true)] $cmd)
docker run --rm --cpus 12 -v "D:\RRC\New folder:/work" -w /work/baselines2d_publication baselines2d:py313 @cmd
