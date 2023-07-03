# Machine-Learning-Based-Long-Jump-Assistant
This repository contains the code files, datasets, videos, and models necessary to run the first iteration of my long jump training assistant

The full paper that details the science and methodology used for this project can be found in the file titled "A Machine Learning Based Long Jump Training Assistant.pdf". The accompanying PowerPoint can be found in the file titled "A Machine Learning Based Long Jump Training Assistant PowerPoint.pptx". The abstract of this paper is as follows:

"The long jump is a highly technical event in track field where the athlete attempts to jump as far
as possible into a sand pit from a running start. Arguably the most important action in this event
is the act of jumping at the end of the run, as this is where the athlete's horizontal energy is
converted vertically. There are many existing papers that mathematically model the long jump
takeoff (the act of jumping) and determine the limb angles, ground angles, and body positions
that optimize the jump. There is a general consensus on the features that most considerably affect
the quality of the takeoff (and therefore the length of the jump). These features are subtle and
nearly impossible to perfectly detect with the human eye. Modern camera technology has
allowed many coaches to routinely film the practice jumps of their athletes, but this method still
relies on human interpretation. This paper proposes a Long Jump training assistant that uses
machine learning to recognize relevant frames of the takeoff, and returns important feature
values to be compared to the objective bio-mechanical model."

More specifically, the final application uses machine learning to recognize 2 important features. The first feature is known as the touchdown frame; this is the frame where the athlete's jump foot first makes contact with the ground. The second feature is known as the takeoff sequence. This is the sequence of frames in which the athlete is jumping off the ground. In other words, the sequence of frames from the touchdown frame to the frame where the athlete's jump leg leaves the ground. Once these 2 features are identified, the program extracts several metrics. The first is the ground angle at touchdown. This is the angle formed from the jump-leg hip, to the jump leg angle, to the horizontal (paralell with the ground). The second metric is the touchdown knee angle. This is the angle formed by the jump-leg hip, the jump-leg knee, and the jump-leg ankle. The third metric extracted is the minimum knee angle of the jump-leg throughout the takeoff sequence. Once these metrics are extracted, they are compared to the biomecahnically most efficient values provided by existing research.

The only Python file necessary to run this application is titled "Final Application.py". The packages necessary to run are included at the beginning of the file. The application utilizes several other files in this repository, and the file-paths will need to be updated if you are trying to run the application on your own computer. <img width="381" alt="image" src="https://github.com/Sobotage/Machine-Learning-Based-Long-Jump-Assistant/assets/110847448/2a5cad82-41b3-4f7a-a4c9-a8eef6e21b9f">
Here is an image of the final application in use.

The file titled "feature_saver.py" shows the general framework I used in order to save feature values from relevant frames into a csv file. One of the files containing feature data is titled "touchdown_frame_feature_data.csv". I also included the two files that contain the code that turned the raw feature data into the models that could be used for machine learning. These are "frame_recognition_model_code.py" and "sequence_recognition_model_code.py". The resulting models from this code are in the files titled "touchdown_recognition_model.pkl" and "sequence recognition_model5.h5" respectively.

Finally, a video of the working application can be found in the file titled "Application_Demo.mov".


