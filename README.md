# BOSCH Mid Prep IIT Jammu
# BOSCH’S AGE AND GENDER DETECTION


Our team built a solution to estimate the gender and age of people from surveillance video feeds (like mall, retail store, hospital etc.). We have developed a pipeline
that has incorporated a facial super resolution technique, that helps our age and gender models in better predictions due to improved fidelity of the detected faces.

All the pretrained weights are now put in models folder and src folder of the submission.

We have tested the code on 3 PC's with and without internet connections, after creating completely new environments.

The code identifies automatically if a GPU is available and runs on it, else it works on CPU.
Yet if the code doesn't work on GPU, we recommend running it completely on CPU.

## Sample Output

Sample Output of Classification Model:

![alt text](https://github.com/Aman-garg-IITian/BOSCH-Mid-Prep-IIT-Jammu/blob/master/output/predictions/grandpa.png?raw=true)


## Deployment

First, extract all the files from the submission folder.

Operating System Requirements: Windows 10 or 11 is preferred (The code might not work in Linux-based OS)

Enviroment Requirement: Use Anaconda to create enviroment.

Python --version >= 3.8 is recommended. 

Python 2 not supported.

If code is showing error then switch to Python 3.8 and re-run the code.

Go into the project/src folder. 


```bash
  cd MP_BO_T1_CODE/src
```
 

Create a virtual environment (preferrably using anaconda), but if you want to run it globally you can skip this step:

```bash
  conda create --name <environment name>
  conda activate <environment name>
```

Install all the dependencies:
```bash
  pip install -r requirements.txt
```
If you get the following error - "no module named "module_name" , use the following command for installing the module:
```bash
  pip install <module_name>
```

IMPORTANT INSTRUCTIONS:

Paste your test video in the 'src' folder (i.e the same folder which has the test file)

Run the following command to run the python file:

```bash
  python test.py --input <test_video_file_name.mp4>
```

Example:

```bash
  python test.py --input Sample_video.mp4
```

A csv file will be generated in the "Final_outputs"(not 'output' one) folder with the same name as the input test video.

You can also view the annotated test video from the "Final_outputs" folder with the same name as the input test video in "mp4" format.






If you want to terminate the program anywhere before its completion, press cmd/ctrl + C. 

The number of frames that get processed before the termination gets
compiled for the desired output in the "Final_outputs folder"


Note that the model does not output the frame number in which no face was detected in the csv file.


