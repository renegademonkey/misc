# Movie Recommender 3000

The files contained in this project document the process of answering two problems:

1) predicting the preferences for specific users towards certain movies
2) predicting how similar any one particular movie is to any other movie

The answers to these two problems can be found in the following project files:

```
answers/evaluation_ratings_answers.csv  
answers/movie_similarity_scores.csv
```

Also included in this project is the code used to generate answers to these problems. 
This code can be re-run to re-create and re-train the ML models needed to make 
predictions based on either existing on novel datasets.

Because of time constraints, the focus of this project was primarily on getting 
the most accurate answers to the problems set. Therefore, the project has NOT as yet 
been optimised for speed or 'ease of use' for non-technical audiences. For re-creating these 
outputs the analysis scripts need to be run individually from the command line (terminal). 

The project was created in the OSX operating system, and while should run on alternative systems, 
this has not been extensively tested. Please use OSX if possible to ensure smooth operation.

**NB: As we are dealing with relatively large datasets, and relatively complex models 
which need to be trained, re-running the code could take up to hours to complete on machines 
with limited CPU/RAM options.**
 

## Getting Started

The files used in the analysis require a working installation of python > 3.6
and several python libraries (see prerequisites). Each file should be run from 
the terminal/commandline in the order in which they appear:

```
01_exploratory_data_analysis.py
02_data_cleaning_and_prep.py
03_model_training_.py
04_answer_scoring.py
```

### Prerequisites

To run these scripts, you will need a working installation of python (3.6 or above) as well as
the following python libraries:

```
pandas==1.0.3
numpy==1.16.4
seaborn==0.10.1
scikit_surprise==1.1.1
matplotlib==3.2.1
nltk==3.5
scikit_learn==0.23.1
surprise==0.1
```

### Installing Prerequisites

##### Python
For information on how to install the latest verison of python on your particular OS, 
please follow the instructions for your OS here: https://realpython.com/installing-python/

##### Downloading code/scripts
To run this recommender, please download the following repository onto your local machine from here:

https://github.com/renegademonkey/misc_pers/archive/master.zip

Once downloaded, extract the files using your favourite compression software. 
Open a command line terminal and navigate to the folder

```
misc_pers-master/movie_recommender/
``` 

##### Installing requisite python libraries
From here, run the following command to install all the pre-requisite libraries needed:

```
pip install -r requirements.txt
```

##### Downloading training data
Finally, download the following three requisite data training files and save into the folder: 

```movie_recommender/model_training/training_data/```

Data files:

[movies_metadata.csv](https://storage.cloud.google.com/ebap-data/technical-test/data-scientist/movies_metadata.csv?hl=fr&organizationId=728448195629&_ga=2.100718145.-1711605022.1537191311)   
[ratings.csv](https://storage.cloud.google.com/ebap-data/technical-test/data-scientist/ratings.csv?hl=fr&organizationId=728448195629&_ga=2.29409023.-1711605022.1537191311)   
[evaluation_ratings.csv](https://storage.cloud.google.com/ebap-data/technical-test/data-scientist/evaluation_ratings.csv?hl=fr&organizationId=728448195629&_ga=2.29409023.-1711605022.1537191311)

NB: you can use new/updated training data, as long as the format (filename, columns, data types etc) is the same as the original training data.


### Running the code

The analysis can be re-run by running each of the 4 scripts individually as needed. N
OTE: the model training and scoring files can take hours to run on different systems.
To run re-run the analysis navigate to the `model_training` folder and type:

```
python 01_exploratory_data_analysis.py

python 02_data_cleaning_and_prep.py

python 03_model_training_.py

python 04_answer_scoring.py
```

NB: These still must be run in order, as later scripts rely on data generated in previous scripts.



## Tests

Test are run throughout the script to ensure data quality and formats. 
In future unit and integration tests could be created.  


## Deployment

The code will need to be refactored considerably to be deployed in a production environment 
to improve both code robustness and latency issues. 

## Author

**Rafael Wlodarski** - [renegademonkey](https://github.com/renegademonkey)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
