# bts-lyrics
BTS Lyrics generator - which generates lyrics from the model trained on all BTS songs. I refered to a project on kaggle which used a Bi-directional LSTM and I fine-tuned the hyperparameters. The model achieved an accuracy of ~83%. As it takes around 9-10 hours to train the model, I have used pickle to store the trained model.
To generate lyrics, run the file - get_output.py
Lyrics generated by LSTM - 
- I miss you so much ears ears know you hyungs on previous you weather dad up you feel was piano years wont are down victory fallen again woo yeah rain me at today again today yeah yeah yeah yeah yeah yeah yeah yeah yeah yeah yeah yeah yeah yeah yeah yeah yeah yeah yeah yeah
- Spring day will come hes shrimps ok man man sorry back us gets here wind 6th superiors bit too hard today yeah yeah yeah breakup yeah breakup love me us tears me� not flower cry yeah yeah yeah breakup yeah yeah breakup yeah breakup young again dream today yeah yeah yeah yeah breakup yeah
- I will free myself of this fake love myself oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh
- I am so sick of this fake love myself laughs but loser before myself again today yeah breakup young out girl more flowed face me never understand away wind it send day even more lives yeah too need oh yeah oh yeah oh yeah oh yeah oh yeah oh yeah oh yeah oh yeah oh yeah oh yeah

___________________________________________________________________________________________________________________________________________________
I have also tried to use Andrej Garpathy's simple transformer code to train a transformer on this dataset. I used tiktoken library by OpenAI to tokenize the lyrics before feeding it to the model. The model achieved a validation loss of   when trained on 500 steps. 
The lyrics generated by the transformer are as follows:


