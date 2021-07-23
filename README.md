Task6_2.ipnyb - containing model building and training with scores

stopwords- repository with stopwords including polish stopwords
 
#Flask application for show results
app.py
tempates -> html file
static/css -> css file

train.py - file to train  and save model 

python test.py --textpath "" --tagpath "" 
(required add paths to test set)

#DEPLOY NOTE
Repo includes Dockerfile for quick build and deploy (Run docker in folder with Dockerfile)

$ docker build --tag <name> .  
$ docker run <name> 

