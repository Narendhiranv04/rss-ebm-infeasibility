"""M4M, E-M4M, I-M4M and DOGAR in MuJoCo and Python (see README.md)."""

# MuJoCo's DLL fails to initialise (WinError 1114) on Windows when pandas has been imported
# first; loading it with the package keeps any import order after `import mamo` safe.
import mujoco  # noqa: F401
