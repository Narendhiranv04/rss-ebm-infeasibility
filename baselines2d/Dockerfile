# Linux runtime for the 2D baselines and the dataset's checker (bxy_teacher / bd_2d need SIGPROF timers).
# Versions follow latent-causal-tamp/requirements.lock.
FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.4.0 scipy==1.16.3 psutil==7.1.3 matplotlib==3.11.2 pillow==12.0.0 pytest==9.1.1
# the checker imports viz.py (matplotlib); build the font cache once instead of in every container
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=/opt/mpl
RUN python -c "import matplotlib.font_manager as f; f._load_fontmanager(try_read_cache=False)"
WORKDIR /work/baselines2d_publication
