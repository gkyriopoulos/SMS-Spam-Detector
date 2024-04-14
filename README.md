A simple SMS spam detector using LLMs.

1) Create a new python virtual enviroment.
2) Incase you have nvidia graphics card just use the requirements.txt
3) Huggingface is required. The installation of Huggingface is system specific (different process for amd/nvdia gpus).
4) Wandb is required. You have to create an account and link your personal access token so you have access to it's api.
5) Run main.py

There are also pretrained models included in case you don't want to wait to train the models. To use them download the pretrained.zip file from the github repo, unzip inside the source folder and simply change PRETRAINED to True in main.py


The specs of the system the project was developed on (and recomended specs) are:

-Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz

-NVIDIA GeForce RTX 3080 Ti

-32GB RAM
