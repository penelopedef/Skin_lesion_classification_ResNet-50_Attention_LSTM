## SKIN LESION DIAGNOSIS USING DEEP LEARNING (Classification with ResNet-50, LSTM, and Attention)
#### Primary Researcher - Penelope DeFreitas 
#### This repository was completed as part of a major research project at the Toronto Metropolitan University in partial fulfillment of the requirements for the degree of Master of Science in the Programme of Data Science and Analytics

## The Problem and Focus Area
The incidence of skin cancer is increasing across North America, with melanoma remaining one of the most aggressive and life-threatening forms. This study evaluated
the impact of class balancing strategies (**no balancing, targeted data augmentation, downsampling, SMOTE-Tomek, and class weighting**) combined with fine-tuning of pre-
trained **ResNet-50 models** for skin lesion classification. In addition, a ResNet-50 model variant that integrated hierarchical diagnosis and attention mechanisms, inspired by Barata et al. (2021), was compared against the baseline model. Using the **ISIC 2017** **dataset** for Phase 1 hyperparameter tuning and the **ISIC 2018 dataset** for further experimentation in Phase 2, model performance was benchmarked using metrics such as accuracy, precision, sensitivity, and AUC-ROC. Results identified the Barata et al. (2021) variant with hybrid class balancing as the most promising, achieving balanced performance across metrics and improving the detection of minority lesion classes.

Furthermore, the integration of **explainable Artificial Intelligence** (**XAI**) added an important dimension to interpreting model decision making, offering significant potential for the real-world clinical adoption of AI-assisted clinical diagnosis.

## Dataset
The dataset was sourced from the ISIC Challenge website - https://challenge.isic-archive.com/data/.

## Required Libraries
Make sure to install the following packages in your Colab/virtual environment:

	• pip install numpy
	• pip install pandas
    • pip install tensorflow==2.12
    • pip install keras==2.12
    • pip install scikit-learn scipy
    • pip install pandas
    • pip install opencv-python-headless
    • pip install matplotlib
	• pip install seaborn
	• pip install h5py


## Experimental Files
    • EDA (Exploratory Data Analysis)
		1. ISIC 2017 EDA - 2017_ISIC_data_EDA.ipynb
		2. ISIC 2018 EDA - 2018_ISIC_data_EDA.ipynb
		
	• Experimental Runs
		- Examples of Earlier Experiments (Models G1.1 - G1.18; ISIC 2017 dataset; ResNet-50 baseline model, Image Augmentation)
			1. Applying Categorical Crossentropy - G1.1.ipynb
			2. Applying Focal Loss - G1.8.ipynb
			3. Applying Weighted Crossentropy - G1.16.ipynb
			
		- Example of Further Fine-tuning Experiments (Models G3.1 - G3.3; ISIC 2017 dataset; ResNet-50 baseline model)
			4. Applying SMOTE-Tomek - G3.3.ipynb
			
		- Example of Applying Augmentation with Downsampling (Models G6.1 - G6.2; ISIC 2018 dataset; Barata et al. (2021) model variant)
			5. G6.2.ipynb
			
		- Example of Applying Class Weighting (Models G8.1 - G8.2; ISIC 2018 dataset; Barata et al. (2021) model variant)
			6. G8.2.ipynb
			
		- Examples of Applying the SMOTE-Tomek with variations (Models G9.1 - G9.3; ISIC 2018 dataset; Barata et al. (2021) model variant)
			7. Downsampling to 50% majority class - G9.1.ipynb
			8. Downsampling to MEL class size - G9.2.ipynb
			9. Augmentation+ downsample 50% majority class+SMOTE-Tomek - G9.3.ipynb
		
	• Grad-CAM (example application)
		1. Grad-CAM.ipynb

## How to Run the Code
This project is optimized for running in Google Colab with GPU acceleration. It is recommended that you run the code in the cloud e.g Google Colab Pro+ (SMOTE-Tomek) and Pro Pro (other class balancing techniques)
	
    1.	Mount your Google Drive 
        from google.colab import drive
        drive.mount('/content/drive')
        
    2.  Set paths for dataset files (update notebook):
        • ISIC 2017 dataset:
			- Training ZIP: /content/...ISIC-2017_Training_Data.zip (replace with the location to your training files on the drive)
			- Testing ZIP: /content/...ISIC-2017_Test_v2_Data.zip (replace with the location to your testing files on the drive)
			- Validation ZIP: /content/...ISIC-2017_Validation_Data.zip (replace with the location to your validation files on the drive)
			- Training CSV: /content/...ISIC-2017_Training_Part3_GroundTruth.csv (replace with the location to your training ground truth file on the drive)
			- Testing CSV: /content/...ISIC-2017_Test_v2_Part3_GroundTruth.csv (replace with the location to your testing ground truth file on the drive)
			- Validation CSV: /content/...ISIC-2017_Validation_Part3_GroundTruth.csv (replace with the location to your validation ground truth file on the drive)
		• ISIC 2018 dataset:
			- Training ZIP: /content/...ISIC2018_Task3_Training_Input.zip (replace with the location to your training files on the drive)
			- Testing ZIP: /content/...ISIC2018_Task3_Test_Input.zip (replace with the location to your testing files on the drive)
			- Validation ZIP: /content/...ISIC2018_Task3_Validation_Input.zip (replace with the location to your validation files on the drive)
			- Training CSV: /content/...ISIC2018_Task3_Training_GroundTruth.csv (replace with the location to your training ground truth file on the drive)
			- Testing CSV: /content/...ISIC2018_Task3_Test_GroundTruth.csv (replace with the location to your testing ground truth file on the drive)
			- Validation CSV: /content/...ISIC2018_Task3_Validation_GroundTruth.csv (replace with the location to your validation ground truth file on the drive)
		• Other Directories for Storage:
			- Deep Learning Models and Results - "/content/drive/MyDrive/Deep Learning/Results" (replace with the location for storing models and results)
			- MODEL_PATH = "/content/drive/MyDrive/Deep Learning/Results/skin_cancer_resnet50_model_best_v9.1-5.h5" (replace with the location of the model of your choice)
			- Grad-CAM results - RESULTS_DIR = /content/drive/MyDrive/Deep Learning/Results/Results_Grad_Cam (replace with the location you intend to store the Grad-CAM results)

    3. To experiment with different hyperparameters (e.g. experiment G1.1):
        • run_experiment(BATCH_SIZE=8, LOSS = 'categorical_crossentropy', LR=1e-4, NUM_classes = 3)
			- replace the BATCH_SIZE, LOSS, LR, NUM_classes with your specific experimental settings
		• NOTE:
			- BATCH_SIZE corresponds to batch size
			- LOSS corresponds to loss function
			- LR corresponds to learning rate
			- NUM_classes corresponds to number of classes in the dataset

    4. Run the code 

## Model Summary
	• Backbone: ResNet-50 pretrained on the ImageNet dataset
	• Attention Module: Channel Attention using Global Average + Max Pooling
	• Sequential Decoder: LSTM with initial hidden state from image features
	• Output: Softmax over 3 classes for ISIC 2018 (Benign, SK, Melanoma) or 7 classes for ISIC 2018 (MSL, NV, BCC, AKIEC, BKL, DF, VASC)
	• Loss Functions: categorical crossentropy, focal loss, weighted crossentropy
	• Class Imbalance Handling: Augmentation, Downsampling, SMOTE-Tomek, Class weighting

## Outputs
	• Saved Models: e.g., best_model_v1.1.h5
	• Reports: Text/json files with precision, recall, BACC, AUC, confusion matrix, confidence/predication results, etc.
	• Metrics-based Visualizations: Accuracy plots, class distribution, sample augmented images
	• Grad-CAM-based Visualizations:
		- raw Grad-CAM heatmap in grayscale (<stem>_gradcam_raw.png)
		- Grad-CAM heatmap as a NumPy array before color mapping (<stem>_gradcam_raw.npy)
		- Grad-CAM heatmap colorized (<stem>_gradcam_raw_color.png)
		- color heatmap blended on top of the image or overlay image (<stem>_gradcam_overlay.png)
		
## Dataset Used
	• ISIC 2017 Skin Lesion Classification Dataset
		- Publicly available [here](https://challenge.isic-archive.com/data/#2017)
		- Includes dermoscopic images of benign, seborrheic keratosis, and melanoma lesions
	• ISIC 2018 Skin Lesion Classification Dataset
		- Publicly available [here](https://challenge.isic-archive.com/data/#2018)
		- Includes dermoscopic images of melanoma, nevus, basal cell carcinoma, actinic keratoses and intraepithelial carcinoma, benign keratosis, dermatofibroma, vascular lesion lesions
