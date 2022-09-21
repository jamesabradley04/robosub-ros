# Computer Vision

Welcome to CV! This will be our final major onboarding task before we move onto regular projects.

The computer vision package listens for images/frames coming from 3 different cameras: left, right, and down. The package 
will then run pre-trained machine learning models on each frame and output bounding boxes for the various objects 
in the frame. These objects could be the gate, buoys, etc. The package will publish to different topics depending 
on which classes are being detected and which cameras are being used.

## Your Task

As an onboarding excercise to familiarize ourselves with how CV is situated within the overall ROS software stack, we will be working with ROS in context of our CV module. You will be helping to build a simplified version of our full CV pipeline, from camera input to outputting to task planning.

### Model Setup

We will begin by completing the ```init_model``` method in our ```detection.py``` script which will take in a ```model_mame``` and load the correct model from its weights file. You may choose to do this in either Pytorch or [Detecto](https://github.com/alankbi/detecto), the Pytorch wrapper we developed a few years ago. We suggest that you use Detecto, as that is the framework we use on our current pipeline, though if you are interested in a challenge, implementing this with Pytorch would be an excellent excercise. Note that your code for the "Making Predictions" and "Publishing" sections will be different (although similar conceptually) depending on which implementation you choose. For the purpose of this onboarding excercise, we will only worry about one model at a time, although generally, you would train a separate object detection model for each task you need computer vision for (gates, buoys, etc.).

In the method ```init_model```, declare a new instance variable called ```predictor``` for either the Pytorch or Detecto model which we will load. You might notice that in the constructor, we have a block of code annotated as ```# Load in model configurations as a dictionary``` which loads in a ```.yaml``` file. If you are confused by this, good, because it is kind of confusing. What this block of code does is that it loads a high-level outline of the model (e.g., the name of the model, the output classes of the model) which will then be used by the rest of our script to load in the weights. Note that this is _not_ the actual weights of the trained model itself.

The ```.yaml``` dictionary is loaded in a variable called ```model_outline```, which you should reference for information like what classes we predict in the model initialization. Since we are only working with one model, for now we can just hardcode an instance variable called ```model_name```. Refer to the actual ```.yaml``` file to see what value ```model_name``` should be.

To load the weights file, look at line 23 and see how we loaded the ```models.yaml``` file. Your code should look similar but not the exact same (look into [```resource_retriever```](http://wiki.ros.org/resource_retriever)). Use ```package://cv/models/{self.model_outline[MODEL_NAME]['weights']``` as your filepath, where ```MODEL_NAME``` will be replaced by the instance variable you created earlier. After reading in the weights file into a file, let's make a call to our ML package's (Pytorch or Detecto) model loader method with the weights file. You may need to look up the official documentation in order to find the write method to call (hint: look for something like ```load```).

Now let's make a call to our init_model method. In our actual code base, we use a slightly different set up (again, because we will use multiple models), but for our purposes, we will just call it when we enable our model in the ```enable_model``` method.

### Making Predictions

Now that we have our method to load a model from a saved weights file onboard the robot, we can start making some predictions with it!

In the ```detect``` method call the model which you have initialized in the ```init_model``` method to make a prediction based on the input image. Note that in the publishing code we provided, we assume that you only make one prediction per image (think about which method to use!). Again, you may need to look up some documentation. If you are using Detecto, check out the source code to the package we linked above to see all of the availible methods in the ```Model``` class.

Note that ```detect``` is actually a callback function which our ROS Subscriber automatically calls everytime we read a new image from the camera feed. See the following code in ```run```:
```Python
rospy.Subscriber(self.camera_feed_topic, Image, self.detect)
```

If everything works, our model should be able to now be initialized and make some inferences! However, as of this point, there is not really a way for us to view what exactly our predictions are. In a sense, our model outputs are "locked in" the ```detection.py``` script as there is no way for us to see what exactly it is outputting from the point of the onboard computer/robot. In order to do so, we will need to first write code to publish our predictions...

### Publishing 

Now that we are able to load up a CV model and make some predictions with it, let's now connect our CV code to the rest of the ROS software stack by publishing a stream of predictions for each image we receive from our camera stream.

Going back to the constructor method, let's instantiate a new ROS Publisher with a topic name based on the model and camera as follows: 
```Python 
f"{self.model_name}/{self.camera}"
```

The publisher constructor will look something like this:
```Python
rospy.Publisher(VAR_NAME, MSG_TYPE, queue_size=10)
```
Complete this line with the appropriate variables.

Note that you will also need to define what type (aka class name) of object we are publishing. For this, look at what we are importing to see if there is anything we can use.

Now that you have created a publisher, we will need to complete the ```publish_predictions``` method. We've provided the code to extract the relevant information to publish from a (Detecto, though outputs from Pytorch would be similar) network; look at what's there and figure out when to call the publisher and with what data.

You are almost done. The last thing we need to do (presuming that our code has no bugs so far) is to make a call to our publishing method. Where would it make the most sense to call ```publish_predictions```?

### Testing

Now that our code is hopefully complete, we can test locally to see if our model can make a prediction on an image. We've provided you with a script called ```test_images.py``` which allows you to simulate a camera stream with a local image (look at the source code for ```test_images.py``` to see/edit what the image should be named and where it should be saved). Read the section below on Running the code with ROS commands as well as the section labeled Examples. You will need to make some changes to the commands (hint: change model name .etc), but the process in terms of the sequence of commands to run should be the same. Don't hesitate to ask us if you get stuck!

### Intermediate Processing (Bonus)

Sometimes, even after our model has finished making predictions, we will still need to "tidy up" our outputs a bit before we publish.

One example of this is filtering our predictions by a confidence threshold so that way suboptimal predictions are removed from the publishing feed after our model has made its inferences. This is important as we don't want our feed to task planning to be filled with bad data (that will need to get filtered out on their end anyway), as this would negatively affect the overall performance of the robot.

This section will be a bit more open-ended, and it is intended that way so you can be creative and do a bit of research on your own. What kind of data do we want to keep? What are some algorithms/filters that can help us "enhance" our data? Do a bit of research on ways to improve the quality of the data we feed to task planning. The sky's the limit!

## Setup

Generally, you would train a separate object detection model for each task you need computer vision for (gates, buoys, etc.). You can then load them as follows:

* Create object detection models and save them as .pth files (see [here](https://github.com/DukeRobotics/documentation/tree/master/cv/training))
* Place these models in the `/models` folder
* Update the `/models/models.yaml` file with each model's details in the following format:

```yaml
<model_name>:  # A name/identifier for your model
  classes: [<class1>, <class2>, ...]  # The classes the model is trained to predict
  topic: <topic_name>  # the base topic name your predictions will be published to
  weights: <file_name>  # the name of your model file
...
```

Example entry for a buoy model:

```yaml
buoy:
  classes: [alien, bat, witch]
  topic: /cv/buoy
  weights: buoy_model.pth
```

Note: To get the model files onto the docker container, you may have to use `scp`. Also, if you come across the following error: 

`URLError: <urlopen error [Errno -3] Temporary failure in name resolution>`

Navigate to [this url](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) 
to manually download the default model file used by the Detecto package. Move this file onto the Docker
container under the directory `/root/.cache/torch/checkpoints/` (do not rename the file). 


## Running

To start up a CV node, run the following command:

```bash
roslaunch cv cv_<camera>.launch
```

Where `<camera>` is one of `left`, `right`, or `down`. 

After starting up a CV node, all models are initially disabled. You can select which model(s) you
want to enable for this camera by using the following service (where `<camera>` is the value you
chose above): 

* `enable_model_<camera>`
  * Takes in the model name (string) and a boolean flag to specify whether to turn the model on or off
  * Returns a boolean indicating whether the attempt was successful
  * Type: custom_msgs/EnableModel
  
Once 1+ models are enabled for a specific node, they listen and publish to topics as described below.

## Topics

#### Listening:

 * `/camera/<camera>/image_raw`
   * The topic that the camera publishes each frame to
   * If no actual camera feed is available, you can simulate one using `roslaunch cv test_images.launch`
   * Type: sensor_msgs/Image

#### Publishing:

* `<topic_name>/<camera>`
  * For each camera frame feed that a model processes, it will publish predictions to this topic  
  * `<topic_name>` is what was specified under `topic` in the `models.yaml` file for each enabled model
    (e.g. the example `buoy` model above might publish to `/cv/buoy/left`)
  * For each detected object in a frame, the model will publish the `xmin`, `ymin`, `xmax`, and `ymax` 
    coordinates (normalized to \[0, 1\], with (0, 0) being the top-left corner), `label` of the object, `score` (a confidence value in the range
    of \[0, 1\]), and the `width` and `height` of the frame. 
    * Note: Only the highest-confidence prediction of each label type is published (e.g. if 5 bounding boxes 
      were predicted for a gate object, only the one with the highest score is chosen)
  * If a model is enabled but detects no objects in a frame, it will publish a message with the label field set to 'none'
  * Type: custom_msgs/CVObject

Note that the camera feed frame rate will likely be greater than the rate at which predictions can 
be generated (especially if more than one model is enabled at the same time), so the publishing rate
could be anywhere from like 0.2 to 10 FPS depending on computing power/the GPU/other factors.  


## Structure

The following are the folders and files in the CV package:

`assets`: Folder with a dummy image to test the CV package on

`launch`: Contains the various launch files for our CV package. There is a general launch file for all the cameras (`cv.launch`), and then there are specific launch files for each camera (`cv_left`, `cv_right`, and `cv_down`). Finally, we have a launch file for our testing script `test_images.launch`

`models`: Contains our pre-trained models and a `.yaml` file that specifies the details of each model (classes predicted, topic name, and the path to the model weights)

`scripts`: This is the "meat" of our package. We have a detection script `detection.py` that will read in images and publish predictions onto a node. We also have a `test_images.py` script that is used for testing our package on a dummy video feed (basically one image repeated over and over). We can simulate different video feeds coming in on the different cameras on our `test_images.py` script.

`CMakeLists.txt`: A text file stating the necessary package dependencies and the files in our package.

`package.xml`: A xml file stating the basic information about the CV package

The CV package also has dependencies in the `core/catkin_ws/src/custom_msgs` folder.

## Examples
To simulate camera feed and then run a model on the feed from the left camera. We'll assume the model we want to run is called 'buoy':
* In one terminal, run `roslaunch cv test_images.launch` to start the script meant to simulate the raw camera feed
* In a new terminal, run `roslaunch cv cv_left.launch` to start the cv node
* In another new terminal, run `rosservice list` and should see the `enable_model_left` service be listed in the terminal
* In the same terminal, run `rosservice call enable_model_left buoy true` to enable the buoy model on the feed coming from the left camera. This model should now be publishing predictions
* To verify that the predictions are being published, you can run `rostopic list`, and you should see both `/camera/left/image_raw` and `/cv/buoy/left` be listed. Then you can run `rostopic echo /cv/buoy/left` and the model predictions should be printed to the terminal