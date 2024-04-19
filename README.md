# Transformer from Scratch

Youtube link: https://www.youtube.com/watch?v=ISNdQcPhsts

## Dataset

https://huggingface.co/datasets/Helsinki-NLP/opus_books

## Prepairing the Environment without using VSCODE

1. Clone the repository in some location (let's supose in /home/$USER):
```bash
cd && git clone https://github.com/FI-UBA/transformer-from-scratch.git && cd transformer-from-scratch
```

2. Build image:
```bash
docker build -t local:fiuba_bumblebee .
```

3. Run the container:
```bash
docker run -it --rm --net=host --name bumblebee -u vscode --gpus all -v ./:/workspaces/transformer-from-scratch -w /workspaces/transformer-from-scratch local:fiuba_bumblebee /bin/bash
```

4. Now, you are running inside the container. From here, you can use like a normal debian-based distribution. So, next step is to create a virtual env called `seminario-env` and activate it:
```bash
python3 -m venv seminario-env && source seminario-env/bin/activate
```

5. Update `pip` and install all required packages:
```bash
pip install -U pip && pip install -r requirements.txt
```

Done!. 

## Running Tensorboard from console

```bash
tensorboard --logdir runs/tmodel --bind_all serve
```

