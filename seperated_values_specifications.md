# Seperated values file specifications


## Making a csv or tsv file that will run in cvasl:

We are BIDS compliant, but have even more specification for how files must be organized to work with CVASL. Do not fear- your file can easily be reorganized to fit our format. If you have any problems with this please contact Dr. Candace Makeda Moore (c.moore@esciencecenter.nl) We w give an examples of how to format files our researcher_interface folder. There the file ['showable_standard.csv'](researcher_interface/sample_sep_values/showable_standard.csv) has the exact format we take. Please note the values for brain area measurements (as well as patient IDs and sex) are made up with in some cases some noise added. Therefore values may not be precisely realistic...the idea is to display the format. 

Please note, all column names should be exactly the same and represent the same values. Therefore we include a dictionary about the column names below:

They need a table like this:

|    column name   | meaning|  units | notes       | data type|
|:---------------------------:|:---------:|:---------:|:---------:|
| 'participant_id',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
| 'participant_id',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'session_id, | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|        
|'run_id', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'age',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'sex',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'site', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'gm_vol', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'wm_vol', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'csf_vol', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'gm_icv_ratio', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'gmwm_icv_ratio',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'wmh_vol', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'wmh_count', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cbf_gm_pvc0', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cbf_gm_pvc2', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cbf_wm_pvc0', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cbf_wm_pvc2', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cbf_aca_pvc0',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cbf_mca_pvc0',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cbf_pca_pvc0',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cbf_aca_pvc2',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cbf_mca_pvc2',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cbf_pca_pvc2',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cov_gm_pvc0', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cov_gm_pvc2', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cov_wm_pvc0', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cov_wm_pvc2', | unique identifier for patient instance| none  | includes patient, run and visit numner   | string|
|'cov_aca_pvc0',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string| 
|'cov_mca_pvc0',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string| 
|'cov_pca_pvc0',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string| 
|'cov_aca_pvc2',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string| 
|'cov_mca_pvc2',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string| 
|'cov_pca_pvc2',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string| 
|'Additional_1',| unique identifier for patient instance| none  | includes patient, run and visit numner   | string| 
|'Additional_2'| unique identifier for patient instance| none  | includes patient, run and visit numner   | string|




 We will soon add a script to check compliance. 