# Use the official Python base image
FROM python:3.8

# Set the working directory
WORKDIR /notebooks

# Install Jupyter notebooks
RUN pip install jupyter

# Install torch, torchvision, PIL, matplotlib, imageio, and scipy
RUN pip install torch torchvision pillow matplotlib imageio scipy

# Expose the Jupyter notebook port
EXPOSE 8888

# Start Jupyter notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
