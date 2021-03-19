# RANZCR-CLiP (Catheter and Line Position Challenge)
## Aims to classify the presence and correct placement of catheter tubes on chest x-rays to save lives of COVID-19 patients.

<img align="center" alt="xray" width="500" src="https://prod-images.static.radiopaedia.org/images/7128572/d28bd594f52d09e864dce9ccbe44d4_gallery.jpg">


# Background
Serious complications can occur as a result of malpositioned lines and tubes in patients. Doctors and nurses frequently use checklists for placement of lifesaving equipment to ensure they follow protocol in managing patients. Yet, these steps can be time consuming and are still prone to human error, especially in stressful situations when hospitals are at capacity.

# Problem
Hospital patients can have catheters and lines inserted during the course of their admission and serious complications can arise if they are positioned incorrectly. Nasogastric tube malpositioning into the airways has been reported in up to 3% of cases, with up to 40% of these cases demonstrating complications [1-3]. Airway tube malposition in adult patients intubated outside the operating room is seen in up to 25% of cases [4,5]. The likelihood of complication is directly related to both the experience level and specialty of the proceduralist. Early recognition of malpositioned tubes is the key to preventing risky complications (even death), even more so now that millions of COVID-19 patients are in need of these tubes and lines.

The gold standard for the confirmation of line and tube positions are chest radiographs. However, a physician or radiologist must manually check these chest x-rays to verify that the lines and tubes are in the optimal position. Not only does this leave room for human error, but delays are also common as radiologists can be busy reporting other scans. Deep learning algorithms may be able to automatically detect malpositioned catheters and lines. Once alerted, clinicians can reposition or remove them to avoid life-threatening complications.

# Radiology Source
The Royal Australian and New Zealand College of Radiologists (RANZCR) is a not-for-profit professional organisation for clinical radiologists and radiation oncologists in Australia, New Zealand, and Singapore. The group is one of many medical organisations around the world (including the NHS) that recognizes malpositioned tubes and lines as preventable. RANZCR is helping design safety systems where such errors will be caught.

# Our Approach
In this competition, we detected the presence and position of catheters and lines on chest x-rays and used machine learning to train and test the model on 40,000 images to categorize a tube that is poorly placed.

# AUC Score
- Private Score: 0.87059
- Public Score: 0.86499

# The Dataset
The dataset has been labelled with a set of definitions to ensure consistency with labelling. The normal category includes lines that were appropriately positioned and did not require repositioning. The borderline category includes lines that would ideally require some repositioning but would in most cases still function adequately in their current position. The abnormal category included lines that required immediate repositioning.

# Successful Application
If successful, your efforts may help clinicians save lives. Earlier detection of malpositioned catheters and lines is even more important as COVID-19 cases continue to surge. Many hospitals are at capacity and more patients are in need of these tubes and lines. Quick feedback on catheter and line placement could help clinicians better treat these patients. Beyond COVID-19, detection of line and tube position will ALWAYS be a requirement in many ill hospital patients.


# Members (Team Brain)
## Team Model Architecture
- Yasmeen Hmaidan
- Maliha Lodi
- Arina Mnatsakanyan

## Team Data
- Stanley Hua
- Whitney Chiu

# Relevant Communities

<img align="left" img src="https://www.internationaldayofradiology.com/app/uploads/2017/08/sponsor-ranzcr.png" width="300">
<img img align="center left" src="https://static.wixstatic.com/media/8f6d2f_a3d012b6a8254c5494ee81979ed84e8b~mv2.png/v1/fill/w_202,h_80,al_c,q_85,usm_0.66_1.00_0.01/uoftaishort_i.webpraw=true" width="300">
<img img align="left" src="https://upload.wikimedia.org/wikipedia/commons/7/7c/Kaggle_logo.png" width="300">

