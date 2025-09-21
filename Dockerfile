# Dockerfile for Marker with XPU support
FROM intel/intel-extension-for-pytorch:2.8.10-xpu

# OS deps commonly used by Marker (OCR, PDF processing, GL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev libleptonica-dev pkg-config \
    libgl1 libglib2.0-0 ca-certificates poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry for dependency management
RUN pip install poetry

# Configure Poetry to install to system environment (not virtualenv)
RUN poetry config virtualenvs.create false

# Install Intel-specific PyTorch packages first to ensure correct versions
RUN python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu && \
    python -m pip install intel-extension-for-pytorch==2.8.10+xpu oneccl_bind_pt==2.8.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# XPU optimization environment variables
ENV ZES_ENABLE_SYSMAN=1 \
    NEOReadDebugKeys=1 \
    ClDeviceGlobalMemSizeAvailablePercent=100 \
    OverrideDefaultFP64Settings=1 \
    IGC_EnableDPEmulation=1 \
    SYCL_CACHE_PERSISTENT=1 \
    SYCL_PI_LEVEL_ZERO_SINGLE_THREAD_MODE=1 \
    SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 \
    IPEX_XPU_ONEDNN_LAYOUT=1 \
    IPEX_XPU_EXECUTION_MODE=performance

# Copy the entire working directory to /opt/app
COPY . /opt/app

# Install Marker dependencies using Poetry, but handle PyTorch separately
WORKDIR /opt/app

# First, export Poetry dependencies to requirements.txt (excluding PyTorch packages)
RUN poetry export -f requirements.txt --output /tmp/requirements.txt --without-hashes || \
    # If export fails, create a minimal requirements.txt with just the missing packages
    echo "starlette>=0.20.0" > /tmp/requirements.txt && \
    echo "fastapi>=0.115.4" >> /tmp/requirements.txt && \
    echo "uvicorn>=0.32.0" >> /tmp/requirements.txt && \
    echo "python-multipart>=0.0.16" >> /tmp/requirements.txt

# Install dependencies with pip, skipping PyTorch packages
RUN pip install -r /tmp/requirements.txt

# Install the package itself
RUN pip install -e .

# Debug: Check if XPU is available after installation
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Has XPU attr:', hasattr(torch, 'xpu')); print('XPU available:', torch.xpu.is_available() if hasattr(torch, 'xpu') else 'N/A')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8015/ || exit 1

EXPOSE 8015

# Launch the Marker server
CMD ["marker_server", "--host", "0.0.0.0", "--port", "8015"]
