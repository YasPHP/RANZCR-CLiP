# RANZCR-CLiP (Catheter and Line Position Challenge)
## Aims to classify the presence and correct placement of catheter tubes on chest x-rays to save lives of COVID-19 patients.


![xray catheter image](https://storage.googleapis.com/kagglesdsdata/competitions/23870/1781260/test/1.2.826.0.1.3680043.8.498.10003659706701445041816900371598078663.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210130%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210130T175044Z&X-Goog-Expires=172799&X-Goog-SignedHeaders=host&X-Goog-Signature=81e4e56a90a41f20837a5233064ab1ad9f92c639d76ba4455dd32a334aed2001199d7540085df00b54848cedad53f47912f9b75bbb2e11c6d01d4f5b87ec599ad171beaefbc0cded9f3ecd836ea173dd553e0193fbae2393aa6051f5e07c8edc60069dea31bcaf6f4b7599956cbd130ad0a11825ec79713d0627f652a7eb7fb61f40e4a6093dbe0cf0e7d2a8cef553557ce0fc323fb6da01f9b2ed2563219d69f5a1e8d6b765d89350d5567fd2485aefd2a49679f741d04fad1a1bfe6d79e0c821b07a00e3a2f08b54b04f25750234e0704dae7286bca32991b7236fb15b9c2837217aa2f71e6871e3b30e4374d78ad70926c24c4f2998243d426702d7270fa9)


# Background
Serious complications can occur as a result of malpositioned lines and tubes in patients. Doctors and nurses frequently use checklists for placement of lifesaving equipment to ensure they follow protocol in managing patients. Yet, these steps can be time consuming and are still prone to human error, especially in stressful situations when hospitals are at capacity.

# Problem
Hospital patients can have catheters and lines inserted during the course of their admission and serious complications can arise if they are positioned incorrectly. Nasogastric tube malpositioning into the airways has been reported in up to 3% of cases, with up to 40% of these cases demonstrating complications [1-3]. Airway tube malposition in adult patients intubated outside the operating room is seen in up to 25% of cases [4,5]. The likelihood of complication is directly related to both the experience level and specialty of the proceduralist. Early recognition of malpositioned tubes is the key to preventing risky complications (even death), even more so now that millions of COVID-19 patients are in need of these tubes and lines.

The gold standard for the confirmation of line and tube positions are chest radiographs. However, a physician or radiologist must manually check these chest x-rays to verify that the lines and tubes are in the optimal position. Not only does this leave room for human error, but delays are also common as radiologists can be busy reporting other scans. Deep learning algorithms may be able to automatically detect malpositioned catheters and lines. Once alerted, clinicians can reposition or remove them to avoid life-threatening complications.

# Radiology Source
The Royal Australian and New Zealand College of Radiologists (RANZCR) is a not-for-profit professional organisation for clinical radiologists and radiation oncologists in Australia, New Zealand, and Singapore. The group is one of many medical organisations around the world (including the NHS) that recognizes malpositioned tubes and lines as preventable. RANZCR is helping design safety systems where such errors will be caught.

# Our Approach
In this competition, we detected the presence and position of catheters and lines on chest x-rays and used machine learning to train and test the model on 40,000 images to categorize a tube that is poorly placed.

# The Dataset
The dataset has been labelled with a set of definitions to ensure consistency with labelling. The normal category includes lines that were appropriately positioned and did not require repositioning. The borderline category includes lines that would ideally require some repositioning but would in most cases still function adequately in their current position. The abnormal category included lines that required immediate repositioning.

# Successful Application
If successful, your efforts may help clinicians save lives. Earlier detection of malpositioned catheters and lines is even more important as COVID-19 cases continue to surge. Many hospitals are at capacity and more patients are in need of these tubes and lines. Quick feedback on catheter and line placement could help clinicians better treat these patients. Beyond COVID-19, detection of line and tube position will ALWAYS be a requirement in many ill hospital patients.


# Members (Team Proj Cog)
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

