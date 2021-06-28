The project requires PyTorch1.6 and other standard Python packages like numpy.

For training -
Run python main.py train "path to the data dir"

For testing -
Run python main.py test "path to the data dir"

For predict -
python main.py predict "path to the data dir" --save_dir "path to the results dir"

For viewing Tensorboard graphs -
tensorboard --logdir=logs
Open the link in a browser or in a new terminal run - 
google-chrome 'http://localhost:6006/'

