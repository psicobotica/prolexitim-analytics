# Prolexitim Experimentation Data

- **Version 1.0 (May 2019)**: Raúl's MPGS TFM Experiment results (Prolexitim TAS-20, Prolexitim NLP). 
- **Version 1.1 (May 2019)**: Clean data from experiments, TAS and NLP merged. API Sentiment (GC,Azure,Watson) added. 
- **Version 1.2 (May 2019)**: TAS Categorical variables added. 
- **Version 1.3 (May 2019)**: Faulty CR end-of-record fixed. 3 duplicates removed. 

# Data Dictionary for Prolexitim Alexithymia Data Set (version 1.3 - merged)

- **RowID**: A unique row identifier. 
- **code**: Anonymous code assigned to each participant (generated by the Prolexitim web apps). 
- **card**: TAT card used in current trial. 
- **hum**: Number of human figures that appear in the corresponding TAT card. 
- **mode**: Language data collection mode (T: transcript from audio, W: written by participant). 
- **time**: Time in milliseconds of narrative building (both audio or written). 
- **G-score**: Google Natural Language API G-score (sentiment analysis). 
- **G-magnitude**: Google Natural Language API G-magnitude (sentiment analysis). 
- **Azure-TA**: Microsoft Azure Text Analytics API score (sentiment analysis). 
- **Text**: Original text (when written) or audio transcript (when spoken). 
- **Text-EN**: English translation of original text. 
- **nlu-sentiment**: IBM Natural Language Understanding API numerical score (sentiment analysis). 
- **nlu-label**: IBM Natural Language Understanding API sentiment label (sentiment analysis). 
- **nlu-joy**: IBM Natural Language Understanding API score for emotion joy (sentiment analysis). 
- **nlu-anger**: IBM Natural Language Understanding API score for emotion anger (sentiment analysis). 
- **nlu-disgust**: IBM Natural Language Understanding API score for emotion disgust (sentiment analysis). 
- **nlu-sadness**: IBM Natural Language Understanding API score for emotion sadness (sentiment analysis). 
- **nlu-fear**: IBM Natural Language Understanding API score for emotion fear (sentiment analysis). 
- **es-len**: Length of original text in Spanish (number of chars). 
- **en-len**: Length of English translation (number of chars). 
- **NLP**: Flag indicating whether or not the participant took the narrative generation exercise. 
- **TAS20**: Global score in the 20-item Toronto Alexithymia Scale. 
- **F1**: F1 subscale of TAS-20.
- **F2**: F2 subscale of TAS-20.
- **F3**: F3 subscale of TAS-20.
- **Tas20Time**: Time in millisenconds the participant took to answer the 20-item TAS questionnaire. 
- **Sex**: Sex as reported by the participant (1-male, 2-female). 
- **Gender**: Gender perception as reported by the participant (1-male, 2-female). 
- **Age**: Current age as reported by the participant (years). 
- **Dhand**: Dominant hand as reported by the participant (1-right, 2-left). 
- **Studies**: Level of studies as reported by the participant (1:None 2:Pri 3:Sec 4:FP 5:Uni 6:Post 7>PhD). 
- **SClass**: Social class as reported by the participant (1:Baja 2:Media 3:Alta). 
- **Siblings**: Total number of siblings including the participant.
- **SibPos**: Participant's position in order of birth.
- **Origin**: Country of origin. 
- **Resid**: Country of residence. 
- **Rtime**: Years living in country of residence.
- **Ethnic**: Ethnic group of the participant (population genetics).
- **Job**: Participant's occupation. 
- **alex-a**: TAS-20 score standard cut-off - Alexithymia categorical variable (Alex, NoAlex, PosAlex). 
- **alex-b**: TAS-20 score lowered cut-off - Alexithymia categorical variable (Alex, NoAlex, PosAlex). 





