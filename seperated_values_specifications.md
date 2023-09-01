# Seperated values file specifications


## Making a csv or tsv file that will run in cvasl:

We are BIDS compliant, but have even more specification for how files must be organized to work with CVASL. Do not fear- your file can easily be reorganized to fit our format. If you have any problems with this please contact Dr. Candace Makeda Moore (c.moore@esciencecenter.nl) We w give an examples of how to format files our researcher_interface folder. There the file ['showable_standard.csv'](researcher_interface/sample_sep_values/showable_standard.csv) has the exact format we take. Please note the values for brain area measurements (as well as patient IDs and sex) are made up with in some cases some noise added. Therefore values may not be precisely realistic...the idea is to display the format. 

Please note, all column names should be exactly the same and represent the same values. Therefore we include a dictionary about the column names below:


|    column name   | meaning|  units | notes     | data type|
|:----------------:|:------:|:------:|:---------:|:---------:|
| 'participant_id'| unique identifier for patient instance| none | includes patient, run and visit number | string|
|'session_id' |identifier for the session| none | visit number in Legacy format| int|        
|'run_id' | unique identifier for instance during session| none | rarely a patient may be imaged multple times, each a run | int|
|'age'| patient's chronological age| years |  exact age best | float|
|'sex'| whether patient male or female| none |  code as 'M' or 'F', intersex patients -> female ('F')  | string|
|'site'| unique identifier for specific machine in specific hospital| none  | none  | string|
|'gm_vol' | total gray matter volume | Liter | bilateral | float|
|'wm_vol', | total white matter volume | Liter | bilateral | float|
|'csf_vol' | cerebrospinal fluid volume | Liter | whole brain | string|
|'gm_icv_ratio' | gray matter to intracranial volume ratio | Liter/Liter  | none | float|
|'gmwm_icv_ratio'| gray matter and white matter to intrcranial volume ratio| Liter/Liter  | none  | float|
|'wmh_vol' | white matter hyperintensities total volume | Liter  | sum bilateral | float|
|'wmh_count' |white matter hyperintensities total count | none  | sum bilateral | int|
|'cbf_gm_pvc0' | cerebral blood flow in grey matter | mL/min/100g | partial volume uncorrected | float|
|'cbf_gm_pvc2' | cerebral blood flow in grey matter | mL/min/100g | partial volume corrected | float|
|'cbf_wm_pvc0' | cerebral blood flow in white matter | mL/min/100g | partial volume uncorrected | float|
|'cbf_wm_pvc2' | cerebral blood flow in white matter | mL/min/100g | partial volume corrected | float|
|'cbf_aca_pvc0'| cerebral blood flow in anterior cerebral artery | mL/min/100g | partial volume uncorrected | float|
|'cbf_mca_pvc0'| cerebral blood flow in middle cerebral artery | mL/min/100g | partial volume uncorrected | float|
|'cbf_pca_pvc0'| cerebral blood flow in posterior cerebral artery | mL/min/100g | partial volume uncorrected | float|
|'cbf_aca_pvc2'| cerebral blood flow in anterior cerebral artery | mL/min/100g | partial volume corrected | float|
|'cbf_mca_pvc2'| cerebral blood flow in middle cerebral artery | mL/min/100g | partial volume corrected | float|
|'cbf_pca_pvc2'| cerebral blood flow in posterior cerebral artery | mL/min/100g  | partial volume corrected | float|
|'cov_gm_pvc0', | spatial coefficient of variation in grey matter| σ/μ | not partial volume corrected | float|
|'cov_gm_pvc2', | spatial coefficient of variation in grey matter| σ/μ | partial volume corrected | float|
|'cov_wm_pvc0', | spatial coefficient of variation in white matter| σ/μ | not partial volume corrected | float|
|'cov_wm_pvc2', | spatial coefficient of variation in white matter| σ/μ | partial volume corrected | float|
|'cov_aca_pvc0'| spatial coefficient of variation in anterior cerebral artery | σ/μ | not partial volume corrected | float| 
|'cov_mca_pvc0'| spatial coefficient of variation in middle cerebral artery | σ/μ | not partial volume corrected | float| 
|'cov_pca_pvc0'| spatial coefficient of variation in posterior cerebral artery| σ/μ | not partial volume corrected | float| 
|'cov_aca_pvc2'| spatial coefficient of variation in anterior cerebral artery| σ/μ | partial volume corrected | float| 
|'cov_mca_pvc2'| spatial coefficient of variation in middle cererbral artery | σ/μ | partial volume corrected | float| 
|'cov_pca_pvc2'| spatial coefficient of variation in posterior cerebral artery | σ/μ | partial volume corrected | float| 
|'Additional_1'| additional value | none  | choice of user   | string, int or float| 
|'Additional_2'| additional value | none  | choice of user   | string, int or float| 




 We will soon add a script to check compliance. 
