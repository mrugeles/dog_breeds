# Dog Breed Classification with Convolutional Neural Networks

The code for this project was created to run in colab in GPU mode. Each notebook has his own dependencies included, so it's only required to upload the notebooks without any aditional support files.

These are the hardware specifications from Colab used for this project:
```
GPU: 1xTesla K80 , having 2496 CUDA cores, compute 3.7,  12GB(11.439GB Usable) GDDR5  VRAM
CPU: 1xsingle core hyper threaded i.e(1 core, 2 threads) Xeon Processors @2.3Ghz (No Turbo Boost) , 45MB Cache
RAM: ~12.6 GB Available
Disk: ~320 GB Available
For every 12hrs or so Disk, RAM, VRAM, CPU cache etc data that is on our alloted virtual machine will get erased
```

## Web application.
Before running the web application, the dataset and bottleneck features need to be downloaded, then the model can be builld with the `build_model.py` utility.

```sh
$ wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
$ unzip -qq dogImages.zip
$ wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz
$ python build_model.py
```

Then the web app can be initialized.

```sh
$ cd app
$ python run.py
```

After initializating the local web server, it will show the following in the console:
```
* Serving Flask app "run" (lazy loading)
* Environment: production
WARNING: Do not use the development server in a production environment.
Use a production WSGI server instead.
* Debug mode: on
* Running on http://0.0.0.0:3001/ (Press CTRL+C to quit)
* Restarting with stat
* Debugger is active!
* Debugger PIN: 315-149-986
 ```
This is how the home page should appear

![Dog app home page](https://raw.githubusercontent.com/mrugeles/mrugeles.github.io/master/images/home_page_dog_app.png)

There's also a notebook that can be run to create the web page model (`build_model.ipynb`). This notebook download its own dependencies.
