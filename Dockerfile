# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tensorflow/tensorflow:2.11.0
# python:3.8-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Suppress TF logs
ENV TF_CPP_MIN_LOG_LEVEL=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser
WORKDIR /app/src

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "solve_game.py", "solve"]
# CMD ["/bin/bash", "ls"]

# Build command
# docker build --tag <image_name> .
# (the cwd should be informaticup-profit/ | the period at the end is importent)

# Run command
# docker run -i --rm --network none --cpus 2.000 --memory 2G --memory-swap 2g <iamge-name>


# Pipe very simple example task
# echo '{"width":40,"height":20,"objects":[{"type":"deposit","x":22,"y":6,"subtype":0,"width":3,"height":3}],"products":[{"type":"product","subtype":0,"resources":[10,0,0,0,0,0,0,0],"points":10}],"turns":100,"time":300}' | docker run -i --rm --network none --cpus 2.000 --memory 2G --memory-swap 2g <image-name>