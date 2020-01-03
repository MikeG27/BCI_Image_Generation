<h1 align = "center"> BCI Image EEG Generator<h1>

Preview 
------------
![Screenshot from 2019-12-23 00-36-08](https://user-images.githubusercontent.com/21131348/71328741-55fa4b00-251c-11ea-8cd3-6369007b43cf.png)


The aim of this project is to build generative neural networks architecture in order to generate images based on EEG signal from human brain. For this particular task author use mobile BCI of [EMOTIV](https://www.emotiv.com/epoc/) company in order to get data from human brain.

Idea 
--------------
![Concept](https://user-images.githubusercontent.com/21131348/71324862-9342e680-24e4-11ea-9600-6d1373a498ad.png)


Tasks:
------------------

0. ~~Perform some reseach about generative models~~
1. ~~**Generate image dataset** as visual stimulus~~ 
2. **Get EEG data** from [Emotiv EPOC Neuroheadset](https://www.emotiv.com/epoc/) 
3. **Train VAE** to generate visual stimuly based on EEG signal 
4. ~~**Train** CNN classifier in order to classify generated images~~
5. **Build web aplication** to test both models 
6. [Optionally] **Build other generative architecture** (GAN)


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

References :
------------
