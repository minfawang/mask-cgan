# Mask CycleGAN

Overview of the work on YouTube: https://youtu.be/VveI1psbmB4

## Prerequisite

Install python dependencies through `pipenv`.

```bash
pipenv install &&
pipenv shell
```

## Start server

```bash
nwb react build client/App.js client/dist/ --title MaskCycleGAN &&
env FLASK_APP=server.py flask run
# Server will run at port 5000
```

## ngrok serving

```bash
ngrok http 5000
```
