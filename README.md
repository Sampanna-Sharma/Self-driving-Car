# Self-driving-Car
Self driving implemented in Asphalt 
Instruction:
  Run asphalt in minimize mode(starting from top left corner of your screen).
  Start the race and run gettrainingdata.py : It records the key stroke and images of the gameplay
  After all the data is recorded run convo_nn.py which makes a neural network model using behaviour cloning(it takes time depending upon        whether your are using GPU or CPU for processing.
  Now run the game, select the track which you have recorded upon
  Run drive.py and let neural network do all the driving for you.
