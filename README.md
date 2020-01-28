<h1 align = "center"> BCI Image EEG Generator (in progress)</h1>

![Screenshot from 2020-01-19 18-16-31](https://user-images.githubusercontent.com/21131348/72685187-df834580-3ae7-11ea-99eb-182c80defa1b.png)



## About The Project

The aim of this project is to generate images from human brain based on EEG signal. In order to do this task author build generative neural network architecture called Variational Autoencoder (VAE) .
For this particular task author get data throught mobile Brain-Computer-Interface of [EMOTIV](https://www.emotiv.com/epoc/) company in order to get data from human brain.

Tasks:
------------------

0. ~~Do reseach about generative models~~
1. ~~**Prepare software for experiment**~~ 
2. ~~**Get EEG data** from [Emotiv EPOC Neuroheadset](https://www.emotiv.com/epoc/)~~
3. ~~Perform data analysis~~
5. ~~Build **preprocessing** pipeline~~
3. ~~**Train VAE** to generate visual stimuly based on EEG signal~~ 
4. ~~**Train** CNN classifier in order to classify generated images~~
5. Get more and better EEG Data
5. **Build web aplication** to test both models 
6. [Optionally] **Build other generative architecture** (GAN)


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.


## Install: 

```bash
git clone "https://github.com/MikeG27/BCI_Image_EEG_Generator.git"
cd BCI_Image_EEG_Generator
pip install -r requirements.txt
```


<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

### Run pipeline 
To run the execution pipeline type in terminal  :
```bash
dvc repro .dvc_files/predict_models.dvc
```

### Visualize pipeline stages 
```bash
dvc pipeline show --ascii .dvc_files/predict_models.dvc
```
![Zrzut ekranu 2020-01-28 o 13 39 33](https://user-images.githubusercontent.com/21131348/73264869-0ae6ee00-41d4-11ea-8488-becc20c613dd.png)


### Visualize pipeline commands
```bash
dvc pipeline show --ascii .dvc_files/predict_models.dvc --commands
```
![Zrzut ekranu 2020-01-28 o 13 39 57](https://user-images.githubusercontent.com/21131348/73264797-e2f78a80-41d3-11ea-84eb-ce3ce77f2106.png)

### Show model metrics
```bash
dvc metrics show
```
![Zrzut ekranu 2020-01-28 o 13 48 14](https://user-images.githubusercontent.com/21131348/73265252-da538400-41d4-11ea-818b-fffaf46ee1a1.png)




<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.


## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Em@il : - [gorskimichal27@gmail.com](gorskimichal27@gmail.com) 
            
Linkedin : [https://www.linkedin.com/in/migorski/](https://www.linkedin.com/in/migorski/)



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python modu
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

