# ausklasser
Code for the paper 'Ausklasser - a classifier for German apprenticeship advertisements' and the model published [here](https://huggingface.co/KKrueger/ausklasser). This code features the experiment pipeline. 

# Running
**Important** pipeline will **NOT** run, if you clone this repository, because data is not public. In the load_data.py script, the actual code has been replaced with pseudo code. This repository is meant for transparency and reproducability. 

The code is stored in the ./src folder. It uses the [GuildAI](https://guild.ai/) framework for experiment run automization and tracking. Having the requirements installed and activated (if using a Venv), running the pipeline as a random search with 100 trials would require to navigate to the folder this repository is stored in locally via the shell and then run:
```console
./ausklasser/src:~$ guild run pipeline --max-trials=100
```
Parameters are defined as Flags in the guild.yml file. Flags can be overwritten (for example for the runs for the final configuration) either by overwriting them in the guild.yml file or directly in the shell:
```console
./ausklasser/src:~$ guild run pipeline lr=0.001 epochs=[4,6] 
```
To run the same parameters several times you can use the 'dummy' flag and overwrite it with the amount of values wanted. 

# Runs 
Guild stores runs in individual folders. It automatically creates a run for each operation (load, train, test and pipeline). All run folders include the code as well all input and output information, including logging. They were exported creating an archive directory. This can be [imported](https://my.guild.ai/t/command-import/82). To access all relevant information, one can filter the runs in commands such as [guild view](https://my.guild.ai/t/command-view/131) or [guild compare](https://my.guild.ai/t/command-compare/77) using filters to access only the runs for the entire pipeline (contains all input and output information neccessary).
 
 ```console
./ausklasser/src:~$ guild view --filter 'operation = pipeline'
```

 Also, via the [python api](https://my.guild.ai/t/python-api/158) run information can be imported into a pandas dataframe. Train folders will also contain tfevent files generated in training.
