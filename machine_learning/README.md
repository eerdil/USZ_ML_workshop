# Tutorial on Machine Learning

**Tutorial prepared by: Prof. Dr. Ender Konukoglu** <br />

In this practical session, participants will learn about basics of machine learning and gain hands on experience with applying available tools. 

This tutorial consists of the following parts:

 * **Part I**: is devoted to learning the basics that will include the following concepts: <br />
  (a) Reading and visualizing data. <br />
  (b) Supervised learning: toy classification and regression with linear models. <br />
  (c) Supervised learning: toy classification and regression with non-linear models. <br />

 * **Part II**: participants will work on real-world data problems: <br />
 (d) automatic diagnosis of Alzheimer’s disease from brain MRI and age regression from brain MRI. <br />

## Setting up your computer

Open a terminal by clicking on "Activities" in the upper left corner of the screen and then search 
for "Terminal" and click on the icon. 

Copy-paste the following lines into your terminal. This will define some essential paths
to interface with our GPU infrastructure. 
Execute the following to reload the settings:

````bash
source /home/excite_01hs20/.bashrc
````

There may be an error message about something being not writable, but this shouldn't matter. 

In the next step we need to download the practical from the git archive to your local machine.
To this end type the following:

````bash
git clone https://git.ee.ethz.ch/krishnch/excite_2020_machine_learning.git 
````

Check the version and path of python using below commands.
If all the steps above are done correctly. You should see the below output.
````bash
which python
expected output: /home/excite_01hs20/miniconda3/bin/python
python -V
expected output: Python 3.6.10 :: Anaconda, Inc.
````

Next switch to the exercise directory by executing:

````bash
cd excite_2020_machine_learning/
````

Now start a jupyter notebook (formely known as iPython notebook) by typing the following 
command:

````bash
jupyter-notebook
````
You should now see some terminal output with a website address in the following format

````bash
http://localhost:8888/?token=<a very long string>
````
The jupyter-notebook should ideally open now in a firefox window.
If it does not open automatically. Open a firefox browser and paste the below url in the address bar.

````bash
http://localhost:8888/
````

That should open a directory structure.
Begin with part I "01a_Reading_data_and_visualization". 

### Contact
For any queries regarding the tutorial please contact:

Krishna Chaitanya (krishna.chaitanya@vision.ee.ethz.ch), PhD Student, ETH Zurich

