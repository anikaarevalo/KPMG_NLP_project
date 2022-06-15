# 🚀Intelligent Automation of Tax Data Collection & Categorisation

## Description ##
**KPMG Belgium**, a leading professional services firm specialising in finance and accounting, wants to ***improve their tax knowledge management system*** that allows their employees to save, access, and load tax-relevant data pertinent to their tasks and activities. To this end, they have contracted us to build **a Machine Learning (ML) solution that would automate the repetitive (and partially manual) tasks of gathering and sorting tax-relevant legal documents published regularly on the Belgian Official Gazettte**. 

We shall present our prototype to the intended **end-users, i.e. KPMG Belgium's Corporate Tax Department**, who stand to benefit from having the monotonous and tedious task of tax data gathering and classification ***automised*** efficiently, cost-effectively, and on demand.

## Usage ##
This is a **plug-and-play digital tool that effectively identitifies and classifies Belgian legal tax documents in the Dutch language**. At this stage of the product cycle development, it is a proof of concept that we can show to our clients who are open to leveraging AI and ML technologies when it comes to the provision of professional financial and accounting services. 

## Data Sources ##

- **Meta data** provided by the client
- **[The Belgian Official Gazette (Dutch)](https://www.ejustice.just.fgov.be/cgi/welcome.pl)** for both training and testing data 
- **Research material on Natural Language Processing (NLP) models** and available libraries were sourced from the following:
  - Latent Dirilicht Allocation (LDA) model optimised by the Mallet algorithm 
    
    https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
  - Training Named Entity Recognition (NER) model in SpaCy
  
    https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/ 

## Visuals ##

                   Our LDA model optimised by the Mallet algorithm showing an example of topic-keyword distribution 
<img width="1000" alt="pipeline 2022-04-21 at 15 49 28" src="https://github.com/anikaarevalo/KPMG_NLP_project/blob/be3c73a5a09b6cc976f9a862facadc7dd37620f7/assets/topic_bubbles.png">


                  Image of the interface of our Deep Learning Solution
                  


## Contributors: "The Tag Collectors" ##

Anika Arevalo (Scrum Master)

Anzeem Arief

Mouad Belayachi

Wouter van de Vijver (Project Manager)

## Timeline ##

12 days

01/06/2022 - 16/06/2022

## Personal situation ##
Our team was able to adequately and successfully train and test two Mallet LDA models in Dutch of two sizes (small and large). As our minimum viable product (MVP), we present to our clients the prototype developed using the small-sized NLP model that can be immediately deployed as a mobile application.
Nonetheless, we also provide a trained large-sized NLP model to anticipate the downstream NLP tasks which KPMG's Advanced Analytics & AI, and Intelligent Automation & New Tech departments could leverage. 

Last but not the least, we are also presenting a prototype of a NER pipeline that can be integrated into this MVP. If given another opportunity and more time, we would like to optimise our prototype further by including this customised (and customisable) pipeline in the design. 
